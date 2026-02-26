/*
 * core - C99 inference engine for the core nano transformer LM
 *
 * Loads a flat binary weight file (exported by scripts/export_weights.py),
 * runs autoregressive generation with temperature sampling.
 *
 * Build: cc -O2 -Wall -std=c99 -o core core.c -lm
 * Usage: ./core <model.bin> "<prompt>" [--temp T] [--tokens N]
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---- data types -------------------------------------------------------- */

typedef struct {
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_layers;
    int ff_dim;
    int max_len;
} Config;

typedef struct {
    float *qkv_w, *qkv_b;   /* [3*embed_dim, embed_dim], [3*embed_dim] */
    float *out_w, *out_b;    /* [embed_dim, embed_dim], [embed_dim]     */
    float *ff1_w, *ff1_b;    /* [ff_dim, embed_dim], [ff_dim]           */
    float *ff2_w, *ff2_b;    /* [embed_dim, ff_dim], [embed_dim]        */
    float *ln1_w, *ln1_b;    /* [embed_dim] each                        */
    float *ln2_w, *ln2_b;    /* [embed_dim] each                        */
} BlockWeights;

typedef struct {
    float *token_embed;      /* [vocab_size, embed_dim]  */
    float *pos_embed;        /* [max_len, embed_dim]     */
    BlockWeights *blocks;    /* [num_layers]             */
    float *ln_f_w, *ln_f_b;  /* [embed_dim] each         */
    float *head;             /* [vocab_size, embed_dim]  */
} Weights;

typedef struct {
    char **tokens;           /* index -> string */
    int *char_to_idx;        /* 256-entry lookup for single-byte chars */
    int vocab_size;
} Vocab;

/* scratch buffers for forward pass */
typedef struct {
    float *x;        /* [max_len, embed_dim] -- sequence hidden states     */
    float *ln_out;   /* [max_len, embed_dim] -- layernorm output scratch   */
    float *qkv_buf;  /* [max_len, 3*embed_dim] -- QKV projections          */
    float *attn_out; /* [max_len, embed_dim] -- attention output per pos   */
    float *att;      /* [num_heads, max_len] -- attention scores (1 query) */
    float *ff_buf;   /* [ff_dim] -- FFN hidden scratch                     */
    float *tmp;      /* [embed_dim] -- general scratch                     */
    float *logits;   /* [vocab_size]                                       */
} RunState;

/* ---- math ops ---------------------------------------------------------- */

/*
 * out = a @ b^T
 * a: [M, K], b: [N, K] (row-major), out: [M, N]
 */
static void matmul(float *out, const float *a, const float *b, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        const float *ai = a + i * K;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            const float *bj = b + j * K;
            for (int k = 0; k < K; k++) {
                sum += ai[k] * bj[k];
            }
            out[i * N + j] = sum;
        }
    }
}

/* out = layernorm(x) with weight w and bias b, dimension n */
static void layernorm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = w[i] * ((x[i] - mean) * inv) + b[i];
    }
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

static float gelu(float x) {
    /* tanh approximation matching PyTorch nn.GELU */
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* ---- forward pass ------------------------------------------------------ */

/*
 * Run full forward pass for tokens[0..pos] (inclusive).
 * Processes ALL positions through each sublayer together (matching PyTorch
 * batch semantics) to guarantee correct attention K,V computation.
 * Writes logits for position `pos` into state->logits.
 */
static void forward(Config *cfg, Weights *w, RunState *state, int *tokens, int pos) {
    int dim = cfg->embed_dim;
    int num_heads = cfg->num_heads;
    int head_dim = dim / num_heads;
    int seq_len = pos + 1;

    /* 1. Embeddings: token + position */
    for (int t = 0; t < seq_len; t++) {
        float *xt = state->x + t * dim;
        float *tok = w->token_embed + tokens[t] * dim;
        float *pe  = w->pos_embed + t * dim;
        for (int i = 0; i < dim; i++) {
            xt[i] = tok[i] + pe[i];
        }
    }

    /* 2. Transformer blocks */
    for (int layer = 0; layer < cfg->num_layers; layer++) {
        BlockWeights *bw = &w->blocks[layer];

        /*
         * --- Sub-layer 1: Pre-norm + Multi-Head Self-Attention ---
         *
         * Step A: LayerNorm all positions, project to QKV
         */
        for (int t = 0; t < seq_len; t++) {
            float *xt  = state->x + t * dim;
            float *lno = state->ln_out + t * dim;
            float *qkv = state->qkv_buf + t * 3 * dim;

            layernorm(lno, xt, bw->ln1_w, bw->ln1_b, dim);
            matmul(qkv, lno, bw->qkv_w, 1, dim, 3 * dim);
            for (int i = 0; i < 3 * dim; i++) {
                qkv[i] += bw->qkv_b[i];
            }
        }

        /*
         * Step B: Compute attention for each position.
         * Q from position t, K/V from positions 0..t (causal mask).
         */
        for (int t = 0; t < seq_len; t++) {
            float *q_all = state->qkv_buf + t * 3 * dim;        /* Q starts at offset 0 */
            float *attn  = state->attn_out + t * dim;
            memset(attn, 0, dim * sizeof(float));

            for (int h = 0; h < num_heads; h++) {
                float *q_h = q_all + h * head_dim;

                /* Dot Q_h with K_h for all positions 0..t */
                for (int s = 0; s <= t; s++) {
                    float *k_s = state->qkv_buf + s * 3 * dim + dim + h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += q_h[d] * k_s[d];
                    }
                    state->att[h * cfg->max_len + s] = score / sqrtf((float)head_dim);
                }

                /* Softmax over 0..t (causal: only these positions exist) */
                softmax(state->att + h * cfg->max_len, t + 1);

                /* Weighted sum of V */
                for (int s = 0; s <= t; s++) {
                    float a = state->att[h * cfg->max_len + s];
                    float *v_s = state->qkv_buf + s * 3 * dim + 2 * dim + h * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        attn[h * head_dim + d] += a * v_s[d];
                    }
                }
            }
        }

        /*
         * Step C: Output projection + residual for all positions
         */
        for (int t = 0; t < seq_len; t++) {
            float *xt   = state->x + t * dim;
            float *attn = state->attn_out + t * dim;

            matmul(state->tmp, attn, bw->out_w, 1, dim, dim);
            for (int i = 0; i < dim; i++) {
                xt[i] += state->tmp[i] + bw->out_b[i];
            }
        }

        /*
         * --- Sub-layer 2: Pre-norm + Feed-Forward ---
         */
        for (int t = 0; t < seq_len; t++) {
            float *xt  = state->x + t * dim;
            float *lno = state->ln_out + t * dim;

            layernorm(lno, xt, bw->ln2_w, bw->ln2_b, dim);

            /* FF1: [1, dim] @ [ff_dim, dim]^T + bias -> GELU */
            matmul(state->ff_buf, lno, bw->ff1_w, 1, dim, cfg->ff_dim);
            for (int i = 0; i < cfg->ff_dim; i++) {
                state->ff_buf[i] = gelu(state->ff_buf[i] + bw->ff1_b[i]);
            }

            /* FF2: [1, ff_dim] @ [dim, ff_dim]^T + bias -> residual */
            matmul(state->tmp, state->ff_buf, bw->ff2_w, 1, cfg->ff_dim, dim);
            for (int i = 0; i < dim; i++) {
                xt[i] += state->tmp[i] + bw->ff2_b[i];
            }
        }
    }

    /* 3. Final layer norm (last position only) */
    float *x_last = state->x + pos * dim;
    layernorm(state->tmp, x_last, w->ln_f_w, w->ln_f_b, dim);

    /* 4. Output head: [1, dim] @ [vocab_size, dim]^T -> [vocab_size] */
    matmul(state->logits, state->tmp, w->head, 1, dim, cfg->vocab_size);
}

/* ---- sampling ---------------------------------------------------------- */

static int sample(float *logits, int n, float temp) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    float inv_temp = 1.0f / temp;
    for (int i = 0; i < n; i++) {
        logits[i] *= inv_temp;
    }
    softmax(logits, n);

    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return n - 1;
}

/* ---- model loading ----------------------------------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static size_t g_mapped_size;  /* for cleanup */

static void *load_model(const char *path, Config *cfg, Weights *w, Vocab *vocab) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open model file");

    struct stat st;
    fstat(fd, &st);
    g_mapped_size = st.st_size;

    void *mapped = mmap(NULL, g_mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) die("mmap failed");

    char *ptr = (char *)mapped;
    char *end = ptr + g_mapped_size;

    if (g_mapped_size < 32) die("model file too small");
    if (memcmp(ptr, "CORE", 4) != 0) die("bad magic (not a core model file)");
    ptr += 4;

    unsigned int version = *(unsigned int *)ptr; ptr += 4;
    if (version != 1) die("unsupported model version");

    unsigned int *cfgp = (unsigned int *)ptr;
    cfg->vocab_size = (int)cfgp[0];
    cfg->embed_dim  = (int)cfgp[1];
    cfg->num_heads  = (int)cfgp[2];
    cfg->num_layers = (int)cfgp[3];
    cfg->ff_dim     = (int)cfgp[4];
    cfg->max_len    = (int)cfgp[5];
    ptr += 6 * 4;

    /* Vocab */
    vocab->vocab_size = cfg->vocab_size;
    vocab->tokens = (char **)malloc(cfg->vocab_size * sizeof(char *));
    vocab->char_to_idx = (int *)malloc(256 * sizeof(int));
    memset(vocab->char_to_idx, -1, 256 * sizeof(int));

    for (int i = 0; i < cfg->vocab_size; i++) {
        if (ptr + 4 > end) die("model file truncated in vocab");
        unsigned int len = *(unsigned int *)ptr; ptr += 4;
        if (ptr + len > end) die("model file truncated in vocab entry");
        vocab->tokens[i] = (char *)malloc(len + 1);
        memcpy(vocab->tokens[i], ptr, len);
        vocab->tokens[i][len] = '\0';
        ptr += len;
        if (len == 1) {
            vocab->char_to_idx[(unsigned char)vocab->tokens[i][0]] = i;
        }
    }

    /* Weight pointers into mmap'd region */
    float *fp = (float *)ptr;
    int dim = cfg->embed_dim;
    int ff  = cfg->ff_dim;
    int vs  = cfg->vocab_size;
    int ml  = cfg->max_len;

    w->token_embed = fp; fp += vs * dim;
    w->pos_embed   = fp; fp += ml * dim;

    w->blocks = (BlockWeights *)malloc(cfg->num_layers * sizeof(BlockWeights));
    for (int b = 0; b < cfg->num_layers; b++) {
        BlockWeights *bw = &w->blocks[b];
        bw->qkv_w = fp; fp += 3 * dim * dim;
        bw->qkv_b = fp; fp += 3 * dim;
        bw->out_w = fp; fp += dim * dim;
        bw->out_b = fp; fp += dim;
        bw->ff1_w = fp; fp += ff * dim;
        bw->ff1_b = fp; fp += ff;
        bw->ff2_w = fp; fp += dim * ff;
        bw->ff2_b = fp; fp += dim;
        bw->ln1_w = fp; fp += dim;
        bw->ln1_b = fp; fp += dim;
        bw->ln2_w = fp; fp += dim;
        bw->ln2_b = fp; fp += dim;
    }

    w->ln_f_w = fp; fp += dim;
    w->ln_f_b = fp; fp += dim;
    w->head   = fp; fp += vs * dim;

    if ((char *)fp > end) die("model file truncated (weights extend past EOF)");

    return mapped;
}

/* ---- tokenization ------------------------------------------------------ */

static int encode(Vocab *vocab, const char *text, int *out, int max_tokens) {
    int n = 0;
    const char *p = text;
    while (*p && n < max_tokens) {
        if (p[0] == '\\' && p[1] == 'n') {
            int idx = vocab->char_to_idx[(unsigned char)'\n'];
            if (idx >= 0) out[n++] = idx;
            p += 2;
            continue;
        }
        if (p[0] == '\\' && p[1] == 't') {
            int idx = vocab->char_to_idx[(unsigned char)'\t'];
            if (idx >= 0) out[n++] = idx;
            p += 2;
            continue;
        }
        int idx = vocab->char_to_idx[(unsigned char)*p];
        if (idx >= 0) {
            out[n++] = idx;
        }
        p++;
    }
    return n;
}

/* ---- main -------------------------------------------------------------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "core - C inference engine for the core nano transformer LM\n"
        "\n"
        "Usage:\n"
        "  %s <model.bin> \"<prompt>\" [options]\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --temp T     Sampling temperature (default: 0.8, 0 = greedy)\n"
        "  --tokens N   Max tokens to generate (default: 256)\n"
        "\n"
        "Examples:\n"
        "  %s models/core.bin \"Q: What is 5+3?\\nA:\"\n"
        "  %s models/core.bin \"Q: Who made you?\\nA:\" --temp 0.5 --tokens 100\n",
        prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = argv[2];
    float temp = 0.8f;
    int max_gen = 256;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_gen = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    srand((unsigned int)time(NULL));

    Config cfg;
    Weights weights;
    Vocab vocab;
    void *mapped = load_model(model_path, &cfg, &weights, &vocab);

    int total_params =
        cfg.vocab_size * cfg.embed_dim +
        cfg.max_len * cfg.embed_dim +
        cfg.num_layers * (3 * cfg.embed_dim * cfg.embed_dim +
                          3 * cfg.embed_dim +
                          cfg.embed_dim * cfg.embed_dim +
                          cfg.embed_dim +
                          cfg.ff_dim * cfg.embed_dim +
                          cfg.ff_dim +
                          cfg.embed_dim * cfg.ff_dim +
                          cfg.embed_dim +
                          4 * cfg.embed_dim) +
        2 * cfg.embed_dim +
        cfg.vocab_size * cfg.embed_dim;

    fprintf(stderr, "model: %d params, %d layers, %d heads, %d dim, %d ff, %d maxlen, %d vocab\n",
            total_params, cfg.num_layers, cfg.num_heads, cfg.embed_dim,
            cfg.ff_dim, cfg.max_len, cfg.vocab_size);

    /* Allocate run state */
    int ml = cfg.max_len;
    int dim = cfg.embed_dim;
    RunState state;
    state.x        = (float *)calloc(ml * dim, sizeof(float));
    state.ln_out   = (float *)malloc(ml * dim * sizeof(float));
    state.qkv_buf  = (float *)malloc(ml * 3 * dim * sizeof(float));
    state.attn_out = (float *)malloc(ml * dim * sizeof(float));
    state.att      = (float *)malloc(cfg.num_heads * ml * sizeof(float));
    state.ff_buf   = (float *)malloc(cfg.ff_dim * sizeof(float));
    state.tmp      = (float *)malloc(dim * sizeof(float));
    state.logits   = (float *)malloc(cfg.vocab_size * sizeof(float));

    /* Tokenize prompt */
    int *tokens = (int *)calloc(ml, sizeof(int));
    int n_prompt = encode(&vocab, prompt, tokens, ml);
    if (n_prompt == 0) die("prompt produced zero tokens");

    /* Print prompt */
    for (int i = 0; i < n_prompt; i++) {
        printf("%s", vocab.tokens[tokens[i]]);
    }
    fflush(stdout);

    /* Autoregressive generation */
    int pos = n_prompt - 1;
    for (int step = 0; step < max_gen; step++) {
        /* Sliding window: if at max length, shift left by half */
        if (pos >= ml - 1) {
            int shift = ml / 2;
            int new_len = pos + 1 - shift;
            memmove(tokens, tokens + shift, new_len * sizeof(int));
            pos = new_len - 1;
        }

        forward(&cfg, &weights, &state, tokens, pos);
        int next = sample(state.logits, cfg.vocab_size, temp);

        /* Token 0 is \x00 -- treat as EOS */
        if (next == 0) break;

        pos++;
        tokens[pos] = next;
        printf("%s", vocab.tokens[next]);
        fflush(stdout);
    }
    printf("\n");

    /* Cleanup */
    free(tokens);
    free(state.x);
    free(state.ln_out);
    free(state.qkv_buf);
    free(state.attn_out);
    free(state.att);
    free(state.ff_buf);
    free(state.tmp);
    free(state.logits);
    free(weights.blocks);
    for (int i = 0; i < vocab.vocab_size; i++) free(vocab.tokens[i]);
    free(vocab.tokens);
    free(vocab.char_to_idx);
    munmap(mapped, g_mapped_size);

    return 0;
}

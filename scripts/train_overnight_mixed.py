#!/usr/bin/env python3
"""Overnight mixed-corpus training run for nous.
Combines all available data (code, reasoning/QA, dialogue, comprehensive).
Periodic checkpoints + sample outputs. Writes structured metrics JSON."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys, json, time, glob
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from transformer import Nous
from tokenizer import CharTokenizer

DATE_TAG = datetime.now().strftime('%Y-%m-%d')
LOG_FILE = f'logs/overnight-train-{DATE_TAG}.log'
METRICS_FILE = f'logs/overnight-metrics-{DATE_TAG}.json'
CHECKPOINT_DIR = 'models/checkpoints'
BEST_MODEL = 'models/overnight_best.pt'
FINAL_MODEL = f'models/overnight_{DATE_TAG}.pt'

# Model config
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 256
MAX_LEN = 128
DROPOUT = 0.1

# Training config - overnight long run
BATCH_SIZE = 32
SEQ_LEN = 64
INITIAL_LR = 1e-3
MIN_LR = 1e-5
NUM_EPOCHS = 200
CHECKPOINT_EVERY = 25
TEST_EVERY = 50

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def save_checkpoint(model, tokenizer, epoch, loss, path):
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'loss': loss, 'vocab_size': tokenizer.vocab_size, 'vocab': tokenizer.char_to_idx
    }, path)

def generate(model, tokenizer, prompt, max_len=80, temperature=0.3):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)[:, -1, :] / temperature
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            generated.append(next_tok)
            text = tokenizer.decode(generated)
            if '\n\nQ:' in text or '\nQ:' in text[len(prompt):]:
                break
            x = torch.cat([x, torch.tensor([[next_tok]])], dim=1)
            if x.size(1) > MAX_LEN:
                x = x[:, -MAX_LEN:]
    model.train()
    return tokenizer.decode(generated)

# ── Load & combine all corpus data ──────────────────────────────────────
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
corpus_files = ['comprehensive.txt', 'jot_code.txt', 'combined_corpus.txt',
                'qa_varied.txt', 'conversational.txt', 'math_comprehensive.txt',
                'sci_fi.txt', 'shakespeare.txt', 'minimal.txt']

combined = []
for f in corpus_files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        with open(path) as fh:
            combined.append(fh.read())

text = '\n\n'.join(combined)
print(f"Mixed corpus: {len(text):,} chars from {len(combined)} files")

# ── Setup ────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

tokenizer = CharTokenizer(text)
dataset = TextDataset(text, tokenizer, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = Nous(vocab_size=tokenizer.vocab_size, embed_dim=EMBED_DIM,
             num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
             ff_dim=FF_DIM, max_len=MAX_LEN, dropout=DROPOUT)

num_params = sum(p.numel() for p in model.parameters())
print(f"Params: {num_params:,} | Vocab: {tokenizer.vocab_size} | Epochs: {NUM_EPOCHS}")

optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR)
criterion = nn.CrossEntropyLoss()

test_prompts = [
    "Q: What is 5+3?\nA:", "Q: What's 7*8?\nA:",
    "Q: What is your name?\nA:", "Q: Who made you?\nA:",
    "Q: print hello world in jot\nA:", "Q: write a function in jot\nA:",
    "Q: What time is it?\nA:", "Q: Tell me a story\nA:",
]

# ── Training ─────────────────────────────────────────────────────────────
metrics = {'epochs': [], 'best_loss': float('inf'), 'total_time_min': 0,
           'num_params': num_params, 'corpus_chars': len(text),
           'corpus_files': len(combined), 'date': DATE_TAG}

log = open(LOG_FILE, 'w')
log.write(f"Overnight training started: {datetime.now()}\nCorpus: {len(text):,} chars\n\n")

best_loss = float('inf')
start = time.time()
model.train()

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    total_loss, n = 0.0, 0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    avg = total_loss / max(n, 1)
    lr = scheduler.get_last_lr()[0]
    dt = time.time() - t0
    scheduler.step()

    msg = f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss: {avg:.4f} | LR: {lr:.6f} | {dt:.1f}s"
    if (epoch + 1) % 10 == 0:
        print(msg)
    log.write(msg + '\n'); log.flush()

    metrics['epochs'].append({'epoch': epoch+1, 'loss': round(avg, 5), 'lr': round(lr, 7), 'time_s': round(dt, 1)})

    if avg < best_loss:
        best_loss = avg
        save_checkpoint(model, tokenizer, epoch+1, avg, BEST_MODEL)

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(model, tokenizer, epoch+1, avg,
                        f'{CHECKPOINT_DIR}/overnight_{epoch+1}.pt')

    if (epoch + 1) % TEST_EVERY == 0:
        samples = []
        log.write(f"\n--- Samples @ epoch {epoch+1} ---\n")
        for p in test_prompts[:4]:
            out = generate(model, tokenizer, p)
            log.write(out + '\n')
            samples.append(out)
        log.write('---\n\n'); log.flush()
        metrics['epochs'][-1]['samples'] = samples

total_time = (time.time() - start) / 60
metrics['best_loss'] = round(best_loss, 5)
metrics['final_loss'] = round(avg, 5)
metrics['total_time_min'] = round(total_time, 1)

save_checkpoint(model, tokenizer, NUM_EPOCHS, avg, FINAL_MODEL)

# Final test
log.write(f"\n{'='*60}\nFINAL TESTS\n{'='*60}\n")
final_samples = []
for p in test_prompts:
    out = generate(model, tokenizer, p, temperature=0.2)
    log.write(out + '\n')
    final_samples.append(out)
    print(out[:120])
metrics['final_samples'] = final_samples

log.write(f"\nDone: {datetime.now()} | {total_time:.1f} min | Best loss: {best_loss:.4f}\n")
log.close()

with open(METRICS_FILE, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ {NUM_EPOCHS} epochs in {total_time:.1f} min | Best loss: {best_loss:.4f}")
print(f"✓ Log: {LOG_FILE}")
print(f"✓ Metrics: {METRICS_FILE}")
print(f"✓ Model: {FINAL_MODEL}")

# nuLLM Quickstart - Get Chatting in 5 Minutes

## Goal
Get a basic conversational AI running on your machine. No API costs.

## Steps

### 1. Install PyTorch (if not already)
```bash
pip3 install torch
```

### 2. Navigate to project
```bash
cd ~/Documents/Code/nuLLM
```

### 3. Run chat
```bash
python3 src/chat.py
```

That's it. The script will:
- Auto-train a tiny model on conversational data (~1 minute)
- Start a chat interface
- Save the model for future use

## Example Session

```
 nuLLM Chat
You: hi
nuLLM: hello there
You: whats up
nuLLM: not much just chilling
You: quit
nuLLM: Bye!
```

## What's Happening

- **Model**: 4-layer transformer, ~500K params
- **Training**: 100 epochs on tiny corpus (~1 min on M-series Mac)
- **Data**: Minimal conversational pairs (greeting, small talk)
- **Quality**: Basic but functional - demonstrates concept

## Limitations

This is a proof-of-concept:
- Small vocabulary (characters only)
- Limited context (128 tokens)
- Trained on ~10 conversation pairs
- Will hallucinate/repeat

## Scaling Up

To improve:
1. Train longer (more epochs)
2. Use word/BPE tokenizer (larger vocab)
3. More training data (conversations, Q&A)
4. Bigger model (more layers/heads)
5. Fine-tune on specific personality

## vs Claude

| Feature | nuLLM (now) | Claude |
|---------|-------------|--------|
| Params | 500K | ~100B+ |
| Training | 1 min | Months |
| Cost | $0 | $20/mo |
| Quality | Basic greetings | Professional |
| Purpose | Learn/experiment | Production |

nuLLM isn't replacing Claude - it's understanding how Claude works.

## Next Steps

- Run `python src/train.py` for better training corpus
- Edit conversation pairs in `src/chat.py`
- Increase model size (embed_dim, layers, heads)
- Try different temperature values for generation

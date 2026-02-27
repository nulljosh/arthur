#!/usr/bin/env python3
"""Train on varied Q&A data"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')
from transformer import Nous
from tokenizer import CharTokenizer

# Load varied data
with open('data/qa_varied.txt') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
print(f"Vocab: {tokenizer.vocab_size} chars")
print(f"Data: {len(text)} chars\n")

# Model config (small but not tiny)
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    ff_dim=128,
    max_len=64,
    dropout=0.1
)

# Prepare data
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(200):
    model.train()
    total_loss = 0
    seq_len = 32
    
    for i in range(0, len(data) - seq_len, seq_len // 2):
        x = data[i:i+seq_len].unsqueeze(0)
        y = data[i+1:i+seq_len+1].unsqueeze(0)
        
        if x.size(1) != y.size(1):
            continue
        
        logits = model(x)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / max(1, (len(data) - seq_len) // (seq_len // 2))
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

# Save
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab': tokenizer.char_to_idx,
    'vocab_size': tokenizer.vocab_size,
    'config': {
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 3,
        'ff_dim': 128,
        'max_len': 64,
        'dropout': 0.1
    }
}
torch.save(checkpoint, 'models/varied.pt')
print("\n✓ Saved to models/varied.pt")

# Test
model.eval()
def test(prompt):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(30):
            if x.size(1) >= 62: x = x[:, -62:]
            logits = model(x)[:, -1, :] / 0.3
            next_token = torch.argmax(logits, dim=-1).item()
            tokens.append(next_token)
            if tokenizer.decode([next_token]) == '\n': break
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(tokens)

print("\n🧪 Testing:")
for q in ["Q: What is your name?\nA: ", "Q: What's 5+5?\nA: ", "Q: Hi\nA: "]:
    print(f"{q.strip()}")
    print(f"   {test(q)}\n")

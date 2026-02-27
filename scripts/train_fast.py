#!/usr/bin/env python3
"""Fast minimal training with verbose logging"""
import torch, torch.nn as nn, sys, os
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Nous
from tokenizer import CharTokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    def __len__(self): return len(self.tokens) - self.seq_len
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

with open('data/minimal.txt') as f: text = f.read()
epochs = 200
print(f"🧠 Training nous - Name + Basic Math")
print(f"📊 Data: {len(text)} chars | Epochs: {epochs}\n")

tokenizer = CharTokenizer(text)
print(f"✓ Vocab: {tokenizer.vocab_size} chars")

dataset = TextDataset(text, tokenizer, seq_len=24)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = Core(vocab_size=tokenizer.vocab_size, embed_dim=64, num_heads=4, num_layers=3, ff_dim=128, max_len=64, dropout=0.1)
params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {params:,} params\n")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

print("🚀 Training...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg = total_loss / len(dataloader)
    if (epoch + 1) % 20 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

torch.save({'model_state_dict': model.state_dict(), 'vocab_size': tokenizer.vocab_size, 'vocab': tokenizer.char_to_idx}, 'models/minimal.pt')
print("\n✓ Saved to models/minimal.pt\n")

print("="*50)
print("🧪 Testing...\n")
model.eval()

def gen(prompt, maxlen=30):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    with torch.no_grad():
        for _ in range(maxlen):
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :] / 0.3, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            decoded = tokenizer.decode(generated)
            if '\nQ:' in decoded[len(prompt):]:
                break
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if x.size(1) > 64:
                x = x[:, -64:]
    return tokenizer.decode(generated)

tests = ["Q: What is your name?\nA:", "Q: What is 5+5?\nA:", "Q: What is 2+2?\nA:", "Q: What is 10+10?\nA:"]
for t in tests:
    result = gen(t)
    answer = result.split('\nA:')[1].split('\nQ:')[0].strip() if '\nA:' in result else result[len(t):].strip()
    print(f"{t} {answer}")

print("\n" + "="*50)

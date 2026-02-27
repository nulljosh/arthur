#!/usr/bin/env python3
"""Ultra-minimal: ONLY 3 QA pairs, heavy repetition"""
import torch, torch.nn as nn, sys, os
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Jore
from tokenizer import CharTokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    def __len__(self): return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

with open('data/ultra_minimal.txt') as f: text = f.read()
print(f"🧠 Ultra-Minimal Training: 3 Q&A pairs only")
print(f"📊 Data: {len(text)} chars\n")

tokenizer = CharTokenizer(text)
print(f"✓ Vocab: {tokenizer.vocab_size} chars")

dataset = TextDataset(text, tokenizer, seq_len=20)
if len(dataset) == 0:
    print("ERROR: Dataset empty!")
    sys.exit(1)
    
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Smaller model for simpler task
model = Core(vocab_size=tokenizer.vocab_size, embed_dim=32, num_heads=2, num_layers=2, ff_dim=64, max_len=32, dropout=0.0)
params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {params:,} params (tiny!)\n")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()

print("🚀 Training 300 epochs...")
model.train()
for epoch in range(300):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg = total_loss / len(dataloader)
    if (epoch + 1) % 30 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d}/300 | Loss: {avg:.6f}")

torch.save({'model_state_dict': model.state_dict(), 'vocab_size': tokenizer.vocab_size, 'vocab': tokenizer.char_to_idx}, 'models/ultra.pt')
print("\n✓ Saved to models/ultra.pt\n")

print("="*60)
print("🧪 Testing with greedy decoding (temp=0.1)...\n")
model.eval()

def gen(prompt):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    with torch.no_grad():
        for _ in range(20):
            logits = model(x)
            # Greedy: pick highest probability
            probs = torch.softmax(logits[:, -1, :] / 0.1, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            generated.append(next_token)
            decoded = tokenizer.decode(generated)
            # Stop at newline after answer
            if '\n' in decoded[len(prompt):]:
                break
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(generated).strip()

tests = ["Q: What is your name?\nA:", "Q: What is 5+5?\nA:", "Q: What is 2+2?\nA:"]
for t in tests:
    result = gen(t)
    # Extract answer
    if '\n' in result:
        answer = result.split('\n')[1] if len(result.split('\n')) > 1 else result
    else:
        answer = result[len(t):].strip()
    print(f"{t} {answer}")

print("\n" + "="*60)
print("✓ Done!")

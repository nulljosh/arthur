#!/usr/bin/env python3
import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_head=2, n_layer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(n_embd, n_head, dim_feedforward=128, batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# Load Shakespeare
corpus = open('data/shakespeare.txt').read()
vocab = sorted(set(corpus))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Encode
data = torch.tensor([char_to_idx[c] for c in corpus], dtype=torch.long)
vocab_size = len(vocab)

# Model
model = SimpleLM(vocab_size, n_embd=32, n_head=2, n_layer=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f'🎭 Training on {len(corpus)} chars, {vocab_size} vocab')
print(f'📊 Model: {sum(p.numel() for p in model.parameters()):,} params')
print()

# Train 200 epochs
for epoch in range(200):
    optimizer.zero_grad()
    logits = model(data[:-1].unsqueeze(0))
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), data[1:])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d}: loss {loss.item():.4f}')

# Save
torch.save({
    'model': model.state_dict(),
    'vocab': vocab,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
}, 'models/shakespeare.pt')
print()
print('✅ Saved to models/shakespeare.pt')

# Generate sample
model.eval()
with torch.no_grad():
    # Seed: "To be"
    seed = "To be"
    context = torch.tensor([char_to_idx[c] for c in seed], dtype=torch.long).unsqueeze(0)
    
    generated = seed
    for _ in range(100):
        logits = model(context)
        probs = torch.softmax(logits[0, -1], dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_idx]
        generated += next_char
        context = torch.cat([context, torch.tensor([[next_idx]])], dim=1)
    
    print()
    print('🎨 Generated sample:')
    print(generated)

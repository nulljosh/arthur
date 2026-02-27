#!/usr/bin/env python3
"""Train nous on comprehensive dataset: math + jot + identity + time/date"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Nous
from tokenizer import CharTokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def save_checkpoint(model, tokenizer, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'vocab_size': tokenizer.vocab_size,
        'vocab': tokenizer.char_to_idx
    }, filename)
    print(f"✓ Checkpoint saved: {filename}")

def test_model(model, tokenizer, prompts, max_len=60, temperature=0.3):
    """Test model with given prompts"""
    model.eval()
    results = []
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        generated = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_len):
                logits = model(x)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                
                # Stop at double newline or next question
                decoded = tokenizer.decode(generated)
                if '\n\nQ:' in decoded or '\nQ:' in decoded[len(prompt):]:
                    break
                
                x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
                if x.size(1) > 128:
                    x = x[:, -128:]
        
        result = tokenizer.decode(generated)
        results.append(result)
    
    model.train()
    return results

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = 'data/comprehensive.txt'
MODEL_FILE = 'models/comprehensive.pt'
LOG_FILE = 'logs/comprehensive_training.log'
CHECKPOINT_DIR = 'models/checkpoints'

# Model config (micro)
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 256
MAX_LEN = 128
DROPOUT = 0.1

# Training config
BATCH_SIZE = 16
SEQ_LEN = 64
INITIAL_LR = 1e-3
MIN_LR = 1e-5
NUM_EPOCHS = 400
CHECKPOINT_EVERY = 40
TEST_EVERY = 25

# ============================================================================
# SETUP
# ============================================================================

print("="*70)
print("TRAINING nous ON COMPREHENSIVE DATASET")
print("Math + Jot + Identity + Time/Date")
print("="*70)

# Load data
with open(DATA_FILE) as f:
    text = f.read()

print(f"\nData: {DATA_FILE}")
print(f"Text length: {len(text):,} chars")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Setup tokenizer and dataset
tokenizer = CharTokenizer(text)
print(f"Vocab size: {tokenizer.vocab_size}")

dataset = TextDataset(text, tokenizer, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    max_len=MAX_LEN,
    dropout=DROPOUT
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params:,}")

# Setup optimizer with cosine annealing scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR
)
criterion = nn.CrossEntropyLoss()

# Test prompts (cover all categories)
test_prompts = [
    # Math
    "Q: What is 5+3?\nA:",
    "Q: What's 7*8?\nA:",
    "Q: Calculate 100/10\nA:",
    # Identity
    "Q: What is your name?\nA:",
    "Q: Who made you?\nA:",
    "Q: What are you?\nA:",
    # Time/Date
    "Q: What time is it?\nA:",
    "Q: What's today's date?\nA:",
    # Jot
    "Q: print hello world in jot\nA:",
    "Q: write a function in jot\nA:",
]

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70 + "\n")

log_file = open(LOG_FILE, 'w')
log_file.write(f"Training started: {datetime.now()}\n")
log_file.write(f"Dataset: math + jot + identity + time/date\n")
log_file.write(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {INITIAL_LR}\n\n")

start_time = time.time()
best_loss = float('inf')

model.train()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count
    current_lr = scheduler.get_last_lr()[0]
    epoch_time = time.time() - epoch_start
    
    # Log progress
    log_msg = f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
    print(log_msg)
    log_file.write(log_msg + "\n")
    log_file.flush()
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_file = f"{CHECKPOINT_DIR}/comprehensive_epoch_{epoch+1}.pt"
        save_checkpoint(model, tokenizer, epoch+1, avg_loss, checkpoint_file)
    
    # Track best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(model, tokenizer, epoch+1, avg_loss, 'models/comprehensive_best.pt')
    
    # Test model periodically
    if (epoch + 1) % TEST_EVERY == 0:
        print("\n" + "-"*70)
        print(f"TESTING AT EPOCH {epoch+1}")
        print("-"*70)
        
        log_file.write(f"\n=== Testing at epoch {epoch+1} ===\n")
        
        for prompt in test_prompts[:4]:  # Test subset during training
            result = test_model(model, tokenizer, [prompt])[0]
            print(f"\n{result}")
            log_file.write(f"\n{result}\n")
        
        print("-"*70 + "\n")
        log_file.write("\n")
        log_file.flush()

# ============================================================================
# FINAL SAVE AND TEST
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time/60:.1f} minutes")
print(f"Best loss: {best_loss:.4f}")

# Save final model
save_checkpoint(model, tokenizer, NUM_EPOCHS, avg_loss, MODEL_FILE)

# Final comprehensive test
print("\n" + "="*70)
print("FINAL TESTING - ALL CATEGORIES")
print("="*70)

log_file.write(f"\n{'='*70}\n")
log_file.write("FINAL TEST RESULTS - ALL CATEGORIES\n")
log_file.write(f"{'='*70}\n\n")

for prompt in test_prompts:
    result = test_model(model, tokenizer, [prompt], temperature=0.2)[0]
    print(f"\n{result}")
    log_file.write(f"\n{result}\n")

log_file.write(f"\nTraining completed: {datetime.now()}\n")
log_file.write(f"Total time: {total_time/60:.1f} minutes\n")
log_file.write(f"Final loss: {avg_loss:.4f}\n")
log_file.write(f"Best loss: {best_loss:.4f}\n")
log_file.close()

print("\n" + "="*70)
print(f"✓ Training log saved to {LOG_FILE}")
print(f"✓ Final model saved to {MODEL_FILE}")
print(f"✓ Best model saved to models/comprehensive_best.pt")
print("="*70)

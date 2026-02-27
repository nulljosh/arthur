#!/usr/bin/env python3
"""
Simple chat interface for jore
Minimal conversational AI - just chat, no tools
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed")
    print("Install: pip3 install torch")
    sys.exit(1)

from transformer import Jore
from tokenizer import CharTokenizer
import pickle


def load_model(model_path, tokenizer_path):
    """Load trained model and tokenizer"""
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    model = Jore(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        ff_dim=256,
        max_len=128,
        dropout=0.0
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, tokenizer


def load_soul():
    """Load Soul.md as system prompt prefix"""
    soul_path = Path(__file__).parent.parent / "Soul.md"
    if soul_path.exists():
        return soul_path.read_text().strip()
    return ""


def generate_response(model, tokenizer, prompt, max_len=50, temperature=0.8):
    """Generate response to prompt"""
    soul = load_soul()
    if soul:
        prompt = f"{soul}\n\nQ: {prompt}\nA:"
    tokens = tokenizer.encode(prompt.lower())
    x = torch.tensor(tokens).unsqueeze(0)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            
            # Stop at sentence end
            if tokenizer.decode([next_token]) in ['.', '!', '?', '\n']:
                break
            
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if x.size(1) > model.max_len:
                x = x[:, -model.max_len:]
    
    response = tokenizer.decode(generated)
    # Extract just the response (after prompt)
    response = response[len(prompt):].strip()
    
    return response if response else "..."


def quick_train():
    """Quick train on minimal conversational data"""
    # Tiny conversational corpus
    corpus = """
hi. hello there.
hello. hey whats up.
whats up. not much just chilling.
how are you. im good thanks.
good morning. morning how are you.
hows it going. going well thanks.
nice to meet you. nice to meet you too.
thank you. youre welcome.
bye. see you later.
see you. bye bye.
"""
    
    print("Quick training conversational model...")
    
    tokenizer = CharTokenizer(corpus)
    
    # Tiny model
    model = Jore(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        ff_dim=256,
        max_len=128,
        dropout=0.1
    )
    
    # Train quickly
    from train import TextDataset, train
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    dataset = TextDataset(corpus, tokenizer, seq_len=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train(model, dataloader, optimizer, 'cpu', epochs=100)
    
    # Save
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/chat_model.pth')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("\n✓ Model trained and saved!")
    return model, tokenizer


def main():
    """Chat loop"""
    model_path = Path('models/chat_model.pth')
    tokenizer_path = Path('models/tokenizer.pkl')
    
    # Train if no model exists
    if not model_path.exists():
        print("No trained model found. Training quickly...")
        model, tokenizer = quick_train()
    else:
        print("Loading trained model...")
        model, tokenizer = load_model(model_path, tokenizer_path)
    
    print("\n" + "="*60)
    print("jore Chat (type 'quit' to exit)")
    print("="*60)
    print("Note: This is a tiny model trained on minimal data.")
    print("Responses will be basic but demonstrate the concept.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("jore: Bye!")
                break
            
            response = generate_response(model, tokenizer, user_input)
            print(f"jore: {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
nuLLM Web UI - Simple Flask interface for text generation
"""

from flask import Flask, render_template, request, jsonify
import torch
from src.model import NanoGPT
from src.tokenizer import CharTokenizer
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/nanoGPT.pt'
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        vocab_size = checkpoint['vocab_size']
        
        tokenizer = CharTokenizer()
        # Rebuild vocab from checkpoint if available
        if 'vocab' in checkpoint:
            tokenizer.char_to_idx = checkpoint['vocab']
            tokenizer.idx_to_char = {v: k for k, v in tokenizer.char_to_idx.items()}
        
        model = NanoGPT(vocab_size=vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        print("  Train a model first: python3 src/train.py --epochs 100")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train first!'}), 400
    
    data = request.json
    prompt = data.get('prompt', 'The')
    length = int(data.get('length', 100))
    temperature = float(data.get('temperature', 0.8))
    
    # Encode prompt
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(context, max_new_tokens=length, temperature=temperature)[0].tolist()
    
    # Decode
    generated_text = tokenizer.decode(generated_ids)
    
    return jsonify({'text': generated_text})

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'vocab_size': len(tokenizer.char_to_idx) if tokenizer else 0
    })

if __name__ == '__main__':
    load_model()
    print("\n🚀 nuLLM Web UI starting...")
    print("   Visit: http://localhost:5000\n")
    app.run(debug=True, port=5000)

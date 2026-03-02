#!/usr/bin/env python3
"""Local API server for Arthur v2 - OpenClaw compatible"""
import sys

import torch
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent / "src"))
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer
from config import ARTHUR_V2_CONFIG, get_last_user_message

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = Path(__file__).parent / "models" / "arthur_v2_epoch1.pt"
TOKENIZER_PATH = Path(__file__).parent / "models" / "bpe_tokenizer_v1.json"

print("Loading Arthur v2...")
model = ArthurV2(**ARTHUR_V2_CONFIG)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

tokenizer = BPETokenizer()
tokenizer.load(str(TOKENIZER_PATH))
print("✓ Model loaded")

@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    prompt = get_last_user_message(data.get('messages', []))

    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            temperature=0.7
        )

    response = tokenizer.decode(output[0].tolist())

    return jsonify({
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': response
            }
        }]
    })

if __name__ == '__main__':
    app.run(port=8888, host='0.0.0.0')

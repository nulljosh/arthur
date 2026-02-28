#!/bin/bash
cd "$(dirname "$0")"
python3 -c "
import sys, json, torch
sys.path.insert(0, 'src')
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer

# Load model
model = ArthurV2(vocab_size=10000)
model.load_state_dict(torch.load('models/arthur_v2_final.pt', map_location='cpu', weights_only=True))
model.eval()

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load('models/bpe_tokenizer_v1.json')

# Read and process
req = json.loads(sys.stdin.read())
prompt = req['messages'][-1]['content']

# For now, just confirm it works
response = f'Arthur v2 received: \"{prompt}\" (Full generation coming soon - model loaded successfully!)'

print(json.dumps({
    'choices': [{
        'message': {
            'role': 'assistant',
            'content': response
        }
    }]
}))
"

#!/usr/bin/env python3
"""Arthur v2 OpenClaw Provider"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer
from config import ARTHUR_V2_CONFIG, get_last_user_message

EOS_TOKEN = 1

class ArthurProvider:
    def __init__(self):
        models_dir = Path(__file__).parent.parent / "models"
        self.model_path = models_dir / "arthur_v2_epoch1.pt"
        self.tokenizer_path = models_dir / "bpe_tokenizer_v1.json"
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        self.model = ArthurV2(**ARTHUR_V2_CONFIG)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        self.model.eval()

        self.tokenizer = BPETokenizer()
        self.tokenizer.load(str(self.tokenizer_path))

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])

        generated = []
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(input_ids)
                next_logits = logits[0, -1, :]

                if temperature > 0:
                    next_logits = next_logits / temperature

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token_id = next_token.item()
                generated.append(token_id)

                if token_id == EOS_TOKEN:
                    break

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated)

if __name__ == "__main__":
    provider = ArthurProvider()

    request_data = json.loads(sys.stdin.read())
    prompt = get_last_user_message(request_data.get("messages", []))
    response = provider.generate(prompt, max_tokens=50)

    print(json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response.strip()
            }
        }]
    }))

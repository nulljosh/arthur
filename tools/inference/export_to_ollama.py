#!/usr/bin/env python3
"""Export Arthur to Ollama format"""
import torch
import json

print("🦙 Exporting Arthur to Ollama...")

# Create Modelfile
modelfile = """FROM ./arthur_v2.bin
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
SYSTEM You are Arthur, a 65M parameter model trained from scratch."""

with open("Modelfile", "w") as f:
    f.write(modelfile)

print("✅ Modelfile created")
print("Run: ollama create arthur -f Modelfile")

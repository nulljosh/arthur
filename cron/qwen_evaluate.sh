#!/bin/bash
# Evaluate Arthur using Qwen as judge

MODEL_PATH="models/arthur_v2.pt"
echo "🧪 Evaluating Arthur with Qwen as judge..."

# Generate test prompts
PROMPTS=("Explain recursion" "Write a haiku" "What is 2+2?")

for prompt in "${PROMPTS[@]}"; do
    echo "Prompt: $prompt"
    
    # Get Arthur's response (mock for now)
    ARTHUR_RESPONSE="Response from Arthur model"
    
    # Ask Qwen to judge
    JUDGMENT=$(echo "Rate this AI response from 1-10: Q: $prompt A: $ARTHUR_RESPONSE" | \
               ollama run qwen2.5:3b 2>/dev/null)
    echo "Qwen judgment: $JUDGMENT"
    echo "---"
done

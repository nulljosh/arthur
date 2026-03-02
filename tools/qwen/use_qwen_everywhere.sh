#!/bin/bash
# Switch all Arthur automation to use local Qwen instead of cloud APIs

echo "🔄 Switching Arthur to use Qwen for everything..."

# 1. Update report.sh to use Qwen
echo "1️⃣ Updating cron/report.sh..."
sed -i '' 's/codex analyze/echo "Analysis:" \&\& ollama run qwen2.5:3b "Analyze these training metrics:"/' cron/report.sh

# 2. Create Qwen-based commit hook
echo "2️⃣ Creating Qwen git hook..."
cat > .git/hooks/prepare-commit-msg << 'HOOK'
#!/bin/bash
# Use Qwen for commit messages
COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

if [ -z "$COMMIT_SOURCE" ]; then
    DIFF=$(git diff --cached --stat)
    MSG=$(echo "Generate a commit message for: $DIFF" | ollama run qwen2.5:3b 2>/dev/null | head -1)
    echo "$MSG" > "$COMMIT_MSG_FILE"
fi
HOOK
chmod +x .git/hooks/prepare-commit-msg

# 3. Create Qwen-based code review daemon
echo "3️⃣ Creating Qwen review daemon..."
cat > daemon/qwen_reviewer.py << 'REVIEWER'
#!/usr/bin/env python3
"""Qwen-based code review daemon for Arthur"""

import subprocess
import time
import os
from pathlib import Path

def review_with_qwen(filepath):
    """Review code changes with Qwen"""
    with open(filepath, 'r') as f:
        code = f.read()[:1000]
    
    prompt = f"Review this code for issues: {code}"
    result = subprocess.run(
        ["ollama", "run", "qwen2.5:3b"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout

def watch_for_changes():
    """Monitor src/ for changes and review with Qwen"""
    src_dir = Path("src")
    last_modified = {}
    
    while True:
        for file in src_dir.glob("*.py"):
            mtime = file.stat().st_mtime
            if file not in last_modified or mtime > last_modified[file]:
                print(f"🔍 Reviewing {file}...")
                review = review_with_qwen(file)
                print(f"Qwen says: {review[:200]}...")
                last_modified[file] = mtime
        time.sleep(10)

if __name__ == "__main__":
    print("🤖 Qwen Review Daemon Started")
    watch_for_changes()
REVIEWER
chmod +x daemon/qwen_reviewer.py

# 4. Update evaluation to use Qwen
echo "4️⃣ Creating Qwen evaluator..."
cat > cron/qwen_evaluate.sh << 'EVAL'
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
EVAL
chmod +x cron/qwen_evaluate.sh

# 5. Create Qwen-powered training assistant
echo "5️⃣ Creating Qwen training assistant..."
cat > daemon/qwen_trainer.sh << 'TRAINER'
#!/bin/bash
# Qwen decides when to stop/continue training

check_training() {
    LOSS=$(tail -1 logs/training.log | grep -oE 'loss: [0-9.]+' | cut -d' ' -f2)
    
    DECISION=$(echo "Current loss: $LOSS. Target: 0.05. Should we continue training? Reply YES or NO only." | \
               ollama run qwen2.5:3b 2>/dev/null)
    
    if [[ "$DECISION" == *"NO"* ]]; then
        echo "🛑 Qwen says stop training"
        pkill -f train_v2.py
    else
        echo "✅ Qwen says continue"
    fi
}

while true; do
    check_training
    sleep 300  # Check every 5 minutes
done
TRAINER
chmod +x daemon/qwen_trainer.sh

echo ""
echo "✅ Arthur now uses Qwen for:"
echo "  - Git commit messages (automatic)"
echo "  - Code review daemon (daemon/qwen_reviewer.py)"
echo "  - Training evaluation (cron/qwen_evaluate.sh)"
echo "  - Training decisions (daemon/qwen_trainer.sh)"
echo "  - Report generation (updated cron/report.sh)"
echo ""
echo "🚀 Start the Qwen reviewer: python3 daemon/qwen_reviewer.py"

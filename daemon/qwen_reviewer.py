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

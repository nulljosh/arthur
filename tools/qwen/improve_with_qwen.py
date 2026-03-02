#!/usr/bin/env python3
"""Use local Qwen to improve Arthur - documentation, tests, and code review"""

import subprocess
import json
import os

def query_qwen(prompt, json_output=False):
    """Query local Qwen model"""
    if json_output:
        prompt += "\nRespond with valid JSON only."
        cmd = ["ollama", "run", "qwen2.5:3b", "--format", "json"]
    else:
        cmd = ["ollama", "run", "qwen2.5:3b"]
    
    result = subprocess.run(cmd, input=prompt, text=True, capture_output=True)
    if json_output:
        try:
            return json.loads(result.stdout.strip())
        except:
            return {"error": "Failed to parse JSON", "raw": result.stdout}
    return result.stdout.strip()

def generate_docstrings():
    """Generate missing docstrings for Python files"""
    print("🔍 Finding files missing docstrings...")
    
    files_to_doc = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    content = f.read()
                    if 'def ' in content and '"""' not in content[:200]:
                        files_to_doc.append(path)
    
    print(f"Found {len(files_to_doc)} files needing docs")
    
    for filepath in files_to_doc[:3]:  # Limit to 3 for demo
        print(f"\n📝 Documenting {filepath}...")
        with open(filepath, "r") as f:
            code = f.read()[:1000]  # First 1000 chars
        
        prompt = f"""Add docstrings to this Python code. Output ONLY the improved code with docstrings:

{code}"""
        
        improved = query_qwen(prompt)
        print(f"Generated docstring preview: {improved[:200]}...")

def generate_tests():
    """Generate unit tests for core functions"""
    print("\n🧪 Generating unit tests...")
    
    test_prompt = """Generate pytest unit tests for a BPE tokenizer with these methods:
- encode(text) -> list of token ids
- decode(ids) -> text string
- train(corpus) -> builds vocabulary

Output a complete test file:"""
    
    tests = query_qwen(test_prompt)
    
    with open("tests/test_tokenizer_qwen.py", "w") as f:
        f.write(tests)
    print("✅ Saved to tests/test_tokenizer_qwen.py")

def suggest_optimizations():
    """Analyze code for optimization opportunities"""
    print("\n⚡ Finding optimization opportunities...")
    
    # Check transformer_v2.py for optimizations
    with open("src/transformer_v2.py", "r") as f:
        code_snippet = f.read()[2000:3000]  # Middle section
    
    prompt = f"""Analyze this transformer code for performance optimizations. Focus on:
1. Memory efficiency
2. Computation speed
3. GPU utilization

Code:
{code_snippet}

Provide 3 specific optimization suggestions:"""
    
    suggestions = query_qwen(prompt)
    print(f"\nOptimization suggestions:\n{suggestions}")

def create_readme_badges():
    """Generate README badges and stats"""
    print("\n🎨 Generating README badges...")
    
    prompt = """Create markdown badges for a machine learning project README:
- Python version 3.10+
- PyTorch
- License: MIT
- Build: passing
- Coverage: 85%
- Model size: 65M params

Output only the markdown badge code:"""
    
    badges = query_qwen(prompt)
    print(f"\nBadges to add to README:\n{badges}")

def main():
    print("🤖 Arthur Improvement Assistant (powered by Qwen)\n")
    print("=" * 50)
    
    # Run all improvements
    generate_docstrings()
    generate_tests()
    suggest_optimizations()
    create_readme_badges()
    
    print("\n✨ Improvement session complete!")
    print("\nNext steps:")
    print("1. Review generated docstrings")
    print("2. Run pytest on new tests")
    print("3. Implement suggested optimizations")
    print("4. Update README with badges")

if __name__ == "__main__":
    main()

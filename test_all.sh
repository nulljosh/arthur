#!/bin/bash
# Test suite for nuLLM conversational training

echo "=== nuLLM Test Suite ==="
echo ""

# Check if training is running
if ps aux | grep -q "[t]rain_conversational"; then
    echo "✓ Training process running"
    ps aux | grep "[t]rain_conversational" | awk '{print "  PID:", $2, "CPU:", $3"%"}'
else
    echo "✗ No training process"
fi

echo ""

# Check for model file
if [ -f "models/conversational.pt" ]; then
    echo "✓ Model file exists"
    ls -lh models/conversational.pt
else
    echo "✗ Model not trained yet"
fi

echo ""

# Check datasets
echo "Dataset sizes:"
for file in data/conversational.txt data/jot_code.txt data/combined_corpus.txt; do
    if [ -f "$file" ]; then
        wc -l "$file" | awk '{print "  " $2 ": " $1 " lines"}'
    fi
done

echo ""

# Test generation if model exists
if [ -f "models/conversational.pt" ]; then
    echo "=== Testing Generation ==="
    source venv/bin/activate
    python3 generate_quick.py "Q: What's jot?\nA:" 2>/dev/null | head -5
fi

echo ""
echo "=== Test Complete ==="

#!/bin/bash
# Retry loop for LLM training - keep trying until success

source venv/bin/activate

LOGFILE="retry_train.log"
MAX_ATTEMPTS=10
ATTEMPT=1

echo "=== LLM Training Retry Loop ===" | tee $LOGFILE
echo "Started: $(date)" | tee -a $LOGFILE
echo "" | tee -a $LOGFILE

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "[Attempt $ATTEMPT/$MAX_ATTEMPTS] $(date)" | tee -a $LOGFILE
    
    # Run training
    python3 -u train_interesting.py >> $LOGFILE 2>&1
    EXIT_CODE=$?
    
    # Check for success
    if [ -f "models/interesting.pt" ]; then
        SIZE=$(stat -f%z models/interesting.pt 2>/dev/null || stat -c%s models/interesting.pt 2>/dev/null)
        if [ "$SIZE" -gt 10000 ]; then
            echo "SUCCESS! Model trained and saved ($SIZE bytes)" | tee -a $LOGFILE
            
            # Generate samples
            echo "" | tee -a $LOGFILE
            echo "Generating samples..." | tee -a $LOGFILE
            tail -50 $LOGFILE | grep -A 100 "Generating samples"
            
            exit 0
        fi
    fi
    
    echo "  Failed (exit $EXIT_CODE), retrying in 10s..." | tee -a $LOGFILE
    sleep 10
    
    ATTEMPT=$((ATTEMPT + 1))
done

echo "FAILED after $MAX_ATTEMPTS attempts" | tee -a $LOGFILE
exit 1

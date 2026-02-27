#!/usr/bin/env python3
import subprocess
import sys
import os

# Add throttle to training by sleeping between batches
# This caps CPU without modifying the training loop

os.chdir('/Users/joshua/Documents/Code/nous')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Run training with output piped through a throttler
proc = subprocess.Popen(
    ['python', '-u', 'src/train.py', '--corpus', 'jot', '--epochs', '200'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

log_file = open('logs/training_capped.log', 'w')
batch_count = 0

for line in proc.stdout:
    log_file.write(line)
    log_file.flush()
    print(line, end='')
    
    # Add 50ms sleep every 10 batches to throttle
    if 'Batch' in line:
        batch_count += 1
        if batch_count % 10 == 0:
            import time
            time.sleep(0.05)  # 50ms throttle per 10 batches

proc.wait()
log_file.close()

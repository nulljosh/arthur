#!/bin/bash
cd /Users/joshua/Documents/Code/nuLLM
source venv/bin/activate
python3 src/train.py --epochs 500 --corpus tiny 2>&1 | tee training_overnight.log

#!/bin/bash
# Wrapper to ensure venv python is used by launchd
set -euo pipefail

cd /Users/joshua/Documents/Code/arthur
exec /Users/joshua/Documents/Code/arthur/venv/bin/python daemon/arthur_watchdog.py

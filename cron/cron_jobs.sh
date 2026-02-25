#!/bin/bash
# Cron jobs for arthur LLM training
# Install: crontab -e, then paste these lines

# Daily 2 AM: Backup latest checkpoint to cold storage
0 2 * * * /Users/joshua/Documents/Code/arthur/cron/backup_checkpoint.sh

# Every 6 hours: Check training health, send status report
0 */6 * * * /Users/joshua/Documents/Code/arthur/cron/health_check.sh

# Weekly Sunday 10 AM: Full dataset audit + optimization
0 10 * * 0 /Users/joshua/Documents/Code/arthur/cron/audit_dataset.sh

# Daily 11:59 PM: Graceful shutdown if training still running (prevent long runs)
59 23 * * * /Users/joshua/Documents/Code/arthur/cron/daily_shutdown.sh

#!/bin/bash
# Sync parquets from EC2 and Mac Mini back to Spark
# Runs via cron every 10 minutes

DEST="/home/matthewmurray/kalshi-forecast/data"
LOG="/tmp/kalshi_sync.log"

echo "$(date) — sync start" >> "$LOG"

# Sync from EC2
rsync -avz --timeout=30 ec2:/home/ec2-user/kalshi-data/ "$DEST/" >> "$LOG" 2>&1 || echo "$(date) — EC2 sync failed (probably offline)" >> "$LOG"

# Sync from Mac Mini
rsync -avz --timeout=30 mini:/tmp/kalshi-data/ "$DEST/" >> "$LOG" 2>&1 || echo "$(date) — Mini sync failed" >> "$LOG"

echo "$(date) — sync done" >> "$LOG"

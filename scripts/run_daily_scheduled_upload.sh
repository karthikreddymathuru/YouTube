#!/usr/bin/env bash
set -e

# Change directory to YouTube workspace
cd /Users/karthik/Desktop/Projects/YouTube

# Setup environment
export PATH="/Users/karthik/Desktop/Projects/YouTube/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

echo "========================================================================"
echo "🎬 [$(date '+%Y-%m-%d %H:%M:%S')] STARTING SCHEDULED DAILY KIDS SHORTS PIPELINE"
echo "========================================================================"

# Run the 6-agent autonomous pipeline with auto-publish
.venv/bin/python main.py agent-run --auto-publish

echo "========================================================================"
echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] SCHEDULED DAILY RUN COMPLETED"
echo "========================================================================"

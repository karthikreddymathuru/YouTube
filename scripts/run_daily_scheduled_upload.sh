#!/usr/bin/env bash
set -e

# Change directory to YouTube workspace dynamically
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

# Setup environment
export PATH="$SCRIPT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export HF_HUB_OFFLINE=1

echo "========================================================================"
echo "🎬 [$(date '+%Y-%m-%d %H:%M:%S')] STARTING SCHEDULED DAILY KIDS SHORTS PIPELINE"
echo "========================================================================"

# Run the 6-agent autonomous pipeline with auto-publish
.venv/bin/python main.py agent-run --auto-publish

echo "========================================================================"
echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] SCHEDULED DAILY RUN COMPLETED"
echo "========================================================================"

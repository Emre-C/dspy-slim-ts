#!/usr/bin/env bash
# Run the LongCoT benchmark on all domains (easy difficulty) to evaluate RLM vs Baseline Predict
# Requires: FIREWORKS_API_KEY exported or in repo-root .env

set -e

# Change to repo root
cd "$(dirname "$0")/.."

export LONGCOT_LM_BACKEND="fireworks"
export LONGCOT_STREAM=1
# Higher token limit is necessary for proper RLM solving
export LONGCOT_MAX_COMPLETION_TOKENS=32768

DOMAINS=("logic" "cs" "chemistry" "chess" "math")
DIFFICULTY="easy"
MAX_QUESTIONS=3 # Scale up as needed for larger runs

echo "Evaluating RLM vs Predict across all LongCoT domains ($DIFFICULTY)..."
echo "Backend: $LONGCOT_LM_BACKEND (Streaming: ON)"
echo "Max Questions per Domain: $MAX_QUESTIONS"
echo "--------------------------------------------------------"

for DOMAIN in "${DOMAINS[@]}"; do
    echo "Running A/B Comparison for domain: $DOMAIN"
    
    npx tsx tools/compare_longcot_predict_rlm.ts \
        --domain "$DOMAIN" \
        --difficulty "$DIFFICULTY" \
        --max "$MAX_QUESTIONS" \
        --no-fallback-score
        
    echo "Completed $DOMAIN"
    echo "--------------------------------------------------------"
done

echo "Evaluation complete! Check the JSONL outputs in tools/longcot/runs/ for full telemetry, latency, and RLM trace turns."

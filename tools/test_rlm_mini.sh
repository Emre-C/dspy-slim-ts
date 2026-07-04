#!/usr/bin/env bash
# Runs the LongCoT mini benchmark strictly evaluating the RLM module.

set -e
cd "$(dirname "$0")/.."

export LONGCOT_LM_BACKEND="fireworks"
export LONGCOT_STREAM=1

echo "Starting LongCoT RLM Mini Benchmark..."
echo "Backend: Kimi K2.5 Turbo (Fireworks API)"
echo "--------------------------------------------------------"

# Run a small mini-bench subset (e.g. 1 question from physics/math/logic/cs)
DOMAINS=("logic" "math" "cs")

for DOMAIN in "${DOMAINS[@]}"; do
    echo "Running RLM test on domain: $DOMAIN"
    
    npx tsx tools/bench_longcot_rlm.ts \
        --runner rlm \
        --domain "$DOMAIN" \
        --difficulty easy \
        --max 1 \
        --max-oracle-calls 10 \
        --no-fallback-score
        
    echo "Completed $DOMAIN"
    echo "--------------------------------------------------------"
done

echo "RLM Mini-bench complete. Review the verify() logs to ensure the Y-Combinator loop solved the queries."

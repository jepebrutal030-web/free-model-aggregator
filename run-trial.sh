#!/usr/bin/env bash
# Run one practice trial locally against the aggregator, then score it.
# Usage: ./run-trial.sh trial-0 [trial-1 ...]
set -u
KIT=~/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit
BOT=~/workspace/makefun/aggregator
for TRIAL in "$@"; do
  echo "=== $TRIAL ==="
  STATE=/tmp/agg-$TRIAL-state
  SCORES=/tmp/agg-$TRIAL-scores
  rm -rf "$STATE" "$SCORES"
  mkdir -p "$STATE" "$SCORES"
  MAKEFUN_ENV_DATA_DIR="$KIT/practice/$TRIAL/input/env" \
  MAKEFUN_STATE_DIR="$STATE" \
  MAKEFUN_ENV_PORT=18091 \
  node "$KIT/program/env/server.mjs" > /tmp/agg-$TRIAL-env.log 2>&1 &
  ENV_PID=$!
  # wait for env ready file
  for _ in $(seq 1 50); do [ -f "$STATE/ready" ] && break; sleep 0.2; done
  if [ ! -f "$STATE/ready" ]; then echo "ENV DID NOT START"; cat /tmp/agg-$TRIAL-env.log; kill $ENV_PID 2>/dev/null; continue; fi
  MAKEFUN_ENV_URL=http://127.0.0.1:18091 \
  MAKEFUN_PROMPT_FILE="$KIT/practice/$TRIAL/input/prompt.txt" \
  MAKEFUN_TRIAL_SEED=0000000000000000000000000000000000000000000000000000000000000000 \
  node "$BOT/main.ts" > /tmp/agg-$TRIAL-bot.log 2>&1 &
  BOT_PID=$!
  # wait for bot to exit (max 120s)
  for _ in $(seq 1 600); do kill -0 $BOT_PID 2>/dev/null || break; sleep 0.2; done
  if kill -0 $BOT_PID 2>/dev/null; then echo "BOT DID NOT EXIT (timeout)"; kill $BOT_PID; fi
  wait $BOT_PID 2>/dev/null; BOT_RC=$?
  echo "bot exit: $BOT_RC"
  sleep 0.5
  kill -TERM $ENV_PID 2>/dev/null
  wait $ENV_PID 2>/dev/null
  if [ -f "$STATE/export/state.json" ]; then
    MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$KIT/practice/$TRIAL/reference" MAKEFUN_SCORES_DIR="$SCORES" \
      node "$KIT/program/scorer/score.mjs"
    echo "scores: $(cat $SCORES/scores.json 2>/dev/null)"
  else
    echo "NO STATE EXPORT"; cat /tmp/agg-$TRIAL-env.log | tail -5
  fi
  echo "--- bot log (tail) ---"; tail -20 /tmp/agg-$TRIAL-bot.log
  echo
done

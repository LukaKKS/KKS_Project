#!/bin/bash
set -e

MODEL=${MODEL:-"gpt-4o-mini"}
RUN_ID=${RUN_ID:-"vico_agent_test"}
EPISODES=${EPISODES:-"2 3 4 5"}
PORT=${PORT:-1072}

ENV_FILE="/Users/giseong/Desktop/multi_agent/vico/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

echo "OPENAI_API_KEY length: ${#OPENAI_API_KEY}"

echo "[run_vico_agent] model=$MODEL, episodes=$EPISODES, run_id=$RUN_ID, port=$PORT"

# Clean up any existing TDW processes on port $PORT (only if they exist)
PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "Killing any existing TDW processes on port $PORT..."
  echo "Killing processes: $PIDS"
  kill -9 $PIDS || true
fi

source /Users/giseong/miniforge3/bin/activate vico

export VICO_REASONER_MODEL=$MODEL

/Users/giseong/miniforge3/envs/vico/bin/python \
  /Users/giseong/Desktop/multi_agent/vico/CoELA/tdw_mat/tdw-gym/challenge.py \
  --agents vico_agent vico_agent \
  --data_path test_env.json \
  --data_prefix dataset/dataset_test/ \
  --experiment_name vico_lite \
  --run_id "$RUN_ID" \
  --communication \
  --eval_episodes $EPISODES \
  --port "$PORT" \
  --screen_size 512 \
  --no_save_img \
  --debug \
  "$@"

# Clean up TDW processes on port $PORT (only if they exist)
PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "Killing any remaining TDW processes on port $PORT..."
  echo "Killing processes: $PIDS"
  kill -9 $PIDS || true
fi

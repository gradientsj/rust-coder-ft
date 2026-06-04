#!/usr/bin/env bash
# Local vLLM endpoint for manual testing of the fine-tuned model.
#
#   serve/vllm_serve.sh <model-dir> [--name qwen3-8b] [--port 8000] [--tp 1]
#                       [--max-len 16384] [--gpu-frac 0.90] [--smoke]
#
# Examples:
#   serve/vllm_serve.sh export/qwen3-8b-ft-fp8 --name rust-coder --smoke
#   serve/vllm_serve.sh export/qwen3-8b-ft --tp 2 --port 8001
#
# Notes (verified against vLLM 0.22 docs):
#  - binds 127.0.0.1 only: local manual testing, never network-exposed
#  - FP8 compressed-tensors checkpoints are AUTO-DETECTED from config.json;
#    same command serves bf16 and fp8 dirs
#  - we fine-tune in NON-thinking format, so thinking is disabled server-wide
#    via --default-chat-template-kwargs; verify responses have no <think> block
#  - chat template comes from the model dir's tokenizer; no flag needed
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:?usage: serve/vllm_serve.sh <model-dir-or-hf-id> [options]}"
shift
NAME="qwen3-8b" PORT=8000 TP=1 MAXLEN=16384 GPUFRAC=0.90 SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)     NAME="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --tp)       TP="$2"; shift 2 ;;
    --max-len)  MAXLEN="$2"; shift 2 ;;
    --gpu-frac) GPUFRAC="$2"; shift 2 ;;
    --smoke)    SMOKE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

LOG="$REPO/serve/vllm_${NAME}_${PORT}.log"
echo "serving $MODEL as '$NAME' on 127.0.0.1:$PORT (tp=$TP, max_len=$MAXLEN) -> $LOG"

# venv bin must be on PATH: vllm's inductor compile shells out to `ninja`
export PATH="$REPO/.venv-serve/bin:$PATH"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPUFRAC" \
  --max-model-len "$MAXLEN" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > "$LOG" 2>&1 &
VLLM_PID=$!
trap '[ "$SMOKE" = 1 ] && kill $VLLM_PID 2>/dev/null || true' EXIT

echo -n "waiting for /health"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo " READY"; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo; echo "vllm died during startup — tail of $LOG:"; tail -20 "$LOG"; exit 1
  fi
  echo -n "."; sleep 2
done

if [ "$SMOKE" = 1 ]; then
  echo "--- smoke: /v1/chat/completions ---"
  # payload via quoted heredoc: backticks in the prompt must NOT be subject
  # to shell command substitution
  PAYLOAD=$(cat <<'JSON'
{"model": "__NAME__", "max_tokens": 256, "messages": [
  {"role": "user",
   "content": "Write a Rust function `fn is_palindrome(s: &str) -> bool` that ignores case. Reply with only the code."}]}
JSON
)
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD/__NAME__/$NAME}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
  echo "--- smoke passed; stopping server (remove --smoke to keep it up) ---"
else
  echo "endpoint:  http://127.0.0.1:$PORT/v1  (model name: $NAME)"
  echo "stop with: kill $VLLM_PID"
  wait $VLLM_PID
fi

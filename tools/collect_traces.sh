#!/usr/bin/env bash
# Сбор трасс для обучения оценщика расхода (tools/fit_cost_model.py).
#
#   tools/collect_traces.sh --seeds "101 102 103" --n 40 --types math,search --max-tokens 15000
#
# Для каждого seed: генерирует набор с разбросом сложности (--spread 1), решает ВСЁ через LLM
# (FAST_PATH=0, иначе дешёвые задачи не попадут в трассы), сверяет ответы и кладёт очищенную трассу
# в data/traces/seed_<N>.jsonl. Останавливается, когда суммарный расход дошёл до --max-tokens.
#
# Ключ берётся только из окружения (LLM_BASE_URL, LLM_API_KEY). Запускать с одной машины:
# на ключе лимит 2 IP.
set -euo pipefail

SEEDS="101 102 103"; N=40; TYPES="coding,data_analysis,math,search"; MAX_TOKENS=15000
OUT="data/traces"; WORK="${TMPDIR:-/tmp}/burnrate_traces"; SPREAD=1; NOVEL=0.3; INFEASIBLE=0.1
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --types) TYPES="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --spread) SPREAD="$2"; shift 2;;
    --novel) NOVEL="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
: "${LLM_BASE_URL:?export LLM_BASE_URL and LLM_API_KEY first}"
: "${LLM_API_KEY:?export LLM_BASE_URL and LLM_API_KEY first}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$OUT" "$WORK"
spent=0
for seed in $SEEDS; do
  left=$(( MAX_TOKENS - spent ))
  if [ "$left" -le 200 ]; then
    echo "предел $MAX_TOKENS токенов достигнут, остановка"; break
  fi
  tasks="$WORK/seed_$seed"; raw="$WORK/seed_$seed.raw.jsonl"
  rm -f "$raw"
  "$PY" "$ROOT/tools/gen_tasks.py" --out "$tasks" --n "$N" --seed "$seed" --spread "$SPREAD" \
      --novel "$NOVEL" --infeasible "$INFEASIBLE" --types "$TYPES" > /dev/null
  echo "seed $seed: $N задач, бюджет на прогон $left"
  FAST_PATH=0 BUDGET="$left" TRAJ_PATH="$raw" "$PY" "$ROOT/run_agent.py" --tasks-dir "$tasks" \
      > "$WORK/seed_$seed.log" 2>&1 || true
  used=$(grep -o '"tokens_spent": [0-9.]*' "$WORK/seed_$seed.log" | tail -1 | awk '{print int($2)}')
  spent=$(( spent + ${used:-0} ))
  "$PY" "$ROOT/tools/strip_traces.py" "$raw" --tasks-dir "$tasks" --group "seed_$seed" \
      --out "$OUT/seed_$seed.jsonl"
  "$PY" "$ROOT/tools/score.py" "$tasks" > "$WORK/seed_$seed.score" || true
  head -6 "$WORK/seed_$seed.score"
  echo "seed $seed: потрачено ${used:-0}, всего $spent из $MAX_TOKENS"
done
echo "готово: трассы в $OUT, потрачено $spent"

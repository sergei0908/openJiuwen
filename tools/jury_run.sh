#!/usr/bin/env bash
# Runs the image the way the jury does, on a copy of a task directory, then scores the answers.
#
#   tools/jury_run.sh examples/tasks
#   IMAGE=hack:latest CPUS=8 MEMORY=16g NETWORK=bridge BUDGET=100000 tools/jury_run.sh /path/to/tasks
#
# The flags follow the command from the regulations (the draft, until the official one is published);
# only the network, CPU count and memory may be overridden: the local machine may have less of them.
# Exit code: non-zero if the run failed, an answer is wrong or an infeasible task got an answer.
set -euo pipefail

TASKS_SRC="${1:?usage: tools/jury_run.sh <tasks-dir>}"
IMAGE="${IMAGE:-hack:latest}"
NETWORK="${NETWORK:-none}"
NPROC="$(nproc)"
CPUS="${CPUS:-$(( NPROC < 4 ? NPROC : 4 ))}"  # the CI runner has 2 CPUs
MEMORY="${MEMORY:-16g}"
TIMEOUT="${TIMEOUT:-1h}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp -r "$TASKS_SRC/." "$WORK/"
# The container runs without capabilities, so it can only write where permissions allow it.
chmod -R a+rwX "$WORK"

MOUNT="$WORK"
if command -v cygpath >/dev/null 2>&1; then  # Git Bash on Windows
    MOUNT="$(cygpath -w "$WORK")"
    export MSYS_NO_PATHCONV=1
fi

timeout -k 1000s "$TIMEOUT" docker run --rm \
    --network "$NETWORK" \
    -e LLM_BASE_URL -e LLM_API_KEY -e BUDGET \
    -v "$MOUNT:/data/tasks:rw" \
    --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
    --cap-drop ALL --security-opt no-new-privileges \
    --cpus "$CPUS" --memory "$MEMORY" --pids-limit 512 \
    --workdir /app --entrypoint python \
    "$IMAGE" \
    /app/run_agent.py --tasks-dir /data/tasks

"${PYTHON:-python3}" "$(dirname "$0")/score.py" "$MOUNT" --strict

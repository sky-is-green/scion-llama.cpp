#!/bin/bash
# Run a GPU command and verify the ACTUAL per-card model buffers.
#
# Canonical HIP order is assumed: ROCm0 = spare (card0), ROCm1 = display (card1).
# The process is killed before generation if the display buffer is not at least
# 2 GiB smaller than the spare buffer. This catches cases where -ts fractions
# do not map to the actual per-layer allocation.
#
# Usage:
#   GPU_RUN_LOG=/tmp/opencode/run.log gpu_run.sh <command> [args...]
#   gpu_run.sh --check-log <path>

set -u

MARGIN_MIB=2048

check_log() {
    local log=$1
    local spare display

    spare=$(grep -oP 'ROCm0 model buffer size =\s*\K[0-9.]+' "$log" 2>/dev/null | sort -n | tail -1)
    display=$(grep -oP 'ROCm1 model buffer size =\s*\K[0-9.]+' "$log" 2>/dev/null | sort -n | tail -1)

    if [ -z "$spare" ] || [ -z "$display" ]; then
        echo "gpu_run: no ROCm0/ROCm1 model buffer report in $log" >&2
        return 2
    fi

    if awk -v s="$spare" -v d="$display" -v m="$MARGIN_MIB" 'BEGIN { exit !(d + m < s) }'; then
        echo "gpu_run: OK spare=${spare} MiB display=${display} MiB (margin >= ${MARGIN_MIB} MiB)" >&2
        return 0
    fi

    echo "gpu_run: REFUSING display=${display} MiB vs spare=${spare} MiB; display must be at least ${MARGIN_MIB} MiB smaller" >&2
    return 3
}

if [ "${1:-}" = "--check-log" ]; then
    check_log "${2:?missing log path}"
    exit $?
fi

LOG=${GPU_RUN_LOG:-/tmp/opencode/gpu-run.log}
: >"$LOG"

"$@" >"$LOG" 2>&1 &
pid=$!

# Wait for the loader to report the per-card buffers.
found=0
for _ in $(seq 1 600); do
    if grep -q "ROCm0 model buffer size" "$LOG" 2>/dev/null && \
       grep -q "ROCm1 model buffer size" "$LOG" 2>/dev/null; then
        found=1
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        exit $?
    fi
    sleep 1
done

if [ "$found" -ne 1 ]; then
    echo "gpu_run: no buffer report after 600 s; killing $pid" >&2
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 2
fi

if ! check_log "$LOG"; then
    kill -TERM "$pid" 2>/dev/null
    sleep 2
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 3
fi

wait "$pid"

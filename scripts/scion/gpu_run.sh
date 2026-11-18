#!/bin/bash
# Run a GPU command and enforce the real Scion display-card policy:
#
#   - ROCm0/ROCm1 are interpreted in the canonical HIP order:
#       ROCm0 = spare (card0), ROCm1 = display (card1)
#   - The display's ACTUAL total VRAM (model + desktop) must stay below
#     DISPLAY_CAP_MIB (default 18 GiB). The display may hold more than the
#     spare, but it must never approach the 19 GB OOM zone.
#   - After the buffers are allocated, if the loader makes no progress for
#     60 s, kill it. This distinguishes a real loader stall from a slow load.
#
# Usage:
#   GPU_RUN_LOG=/tmp/opencode/run.log gpu_run.sh <command> [args...]
#   gpu_run.sh --check-log <path>

set -u

DISPLAY_CAP_MIB=${DISPLAY_CAP_MIB:-18432}   # 18 GiB
DESKTOP_RESERVE_MIB=${DESKTOP_RESERVE_MIB:-2048}
SYSFS_DISPLAY=${SYSFS_DISPLAY:-card1}
VRAM_USED=/sys/class/drm/$SYSFS_DISPLAY/device/mem_info_vram_used

buffer_mib() {
    grep -oP "$1 model buffer size =\s*\K[0-9.]+" "$2" 2>/dev/null | sort -n | tail -1
}

check_log() {
    local log=$1
    local dbuf

    dbuf=$(buffer_mib ROCm1 "$log")
    if [ -z "$dbuf" ]; then
        echo "gpu_run: no ROCm1 model buffer report in $log" >&2
        return 2
    fi

    if awk -v d="$dbuf" -v r="$DESKTOP_RESERVE_MIB" -v c="$DISPLAY_CAP_MIB" \
            'BEGIN { exit !(d + r < c) }'; then
        echo "gpu_run: OK display buffer=${dbuf} MiB + reserve=${DESKTOP_RESERVE_MIB} MiB < cap=${DISPLAY_CAP_MIB} MiB" >&2
        return 0
    fi

    echo "gpu_run: REFUSING display buffer=${dbuf} MiB + reserve=${DESKTOP_RESERVE_MIB} MiB >= cap=${DISPLAY_CAP_MIB} MiB" >&2
    return 3
}

if [ "${1:-}" = "--check-log" ]; then
    check_log "${2:?missing log path}"
    exit $?
fi

LOG=${GPU_RUN_LOG:-/tmp/opencode/gpu-run.log}
: >"$LOG"

# Run the child in its own process group (job control) so the cleanup trap below
# can take down the whole group.
set -m
"$@" >"$LOG" 2>&1 &
pid=$!
set +m

# The child gets its own process group so a timeout or Ctrl-C on this script takes
# the model down with it. Without this, a killed run leaves llama-cli holding VRAM
# and the next run fails to allocate (or silently drops to a different layout).
trap 'kill -TERM -"$pid" 2>/dev/null; sleep 1; kill -KILL -"$pid" 2>/dev/null' EXIT INT TERM

# Wait for the loader's per-card buffer report.
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

# Hard cap: the display's actual total VRAM, including the desktop.
used_bytes=$(cat "$VRAM_USED" 2>/dev/null || echo 0)
used_mib=$((used_bytes / 1024 / 1024))
if awk -v u="$used_mib" -v c="$DISPLAY_CAP_MIB" 'BEGIN { exit !(u < c) }'; then
    echo "gpu_run: OK display total=${used_mib} MiB < cap=${DISPLAY_CAP_MIB} MiB" >&2
else
    echo "gpu_run: REFUSING display total=${used_mib} MiB >= cap=${DISPLAY_CAP_MIB} MiB; killing $pid" >&2
    kill -TERM "$pid" 2>/dev/null
    sleep 2
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 3
fi

# 60 s load-progress check: a real stall produces no further log line.
progress=0
for _ in $(seq 1 60); do
    if grep -qE "llama_model_load: model loaded|build_lora_branch|llama_context: constructing|prompt eval|eval time|print_timing" "$LOG" 2>/dev/null; then
        progress=1
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    sleep 1
done

if [ "$progress" -ne 1 ] && kill -0 "$pid" 2>/dev/null; then
    echo "gpu_run: no load progress for 60 s; killing $pid" >&2
    kill -TERM "$pid" 2>/dev/null
    sleep 2
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit 4
fi

wait "$pid"

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

vram_mib() {
    echo $(( $(cat "$VRAM_USED" 2>/dev/null || echo 0) / 1024 / 1024 ))
}

# Enforce the display cap for the whole run, starting now. This cannot wait for the
# loader: a tensor-parallel (meta) buffer prints one total instead of the per-card
# report the check below looks for, and the cap has to hold either way. It also
# covers allocations that grow later (KV and compute buffers grow with context).
(
    while kill -0 "$pid" 2>/dev/null; do
        cur=$(vram_mib)
        if [ "$cur" -ge "$DISPLAY_CAP_MIB" ]; then
            echo "gpu_run: display total=${cur} MiB >= cap=${DISPLAY_CAP_MIB} MiB; killing $pid" >&2
            kill -TERM -"$pid" 2>/dev/null
            sleep 2
            kill -KILL -"$pid" 2>/dev/null
            exit 3
        fi
        sleep 2
    done
) &
cap_mon=$!
echo "gpu_run: display cap monitor active (cap=${DISPLAY_CAP_MIB} MiB)" >&2

# Wait for the loader's per-card buffer report, for the informational check below.
# A tensor-parallel run prints one "Meta() model buffer size" instead and never the
# per-card lines, so that line is the signal to stop waiting. The monitor above is
# what enforces the cap; this is only the message.
found=0
for _ in $(seq 1 600); do
    if grep -q "ROCm0 model buffer size" "$LOG" 2>/dev/null && \
       grep -q "ROCm1 model buffer size" "$LOG" 2>/dev/null; then
        found=1
        break
    fi
    if grep -q "Meta() model buffer size" "$LOG" 2>/dev/null; then
        found=2
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        found=3
        break
    fi
    sleep 1
done

# A run without -v writes little, so its buffered lines may only reach the file when
# the process exits. If the loop ended because the child went away, check once more.
if [ "$found" -eq 0 ] || [ "$found" -eq 3 ]; then
    if grep -q "Meta() model buffer size" "$LOG" 2>/dev/null; then
        found=2
    elif grep -q "ROCm0 model buffer size" "$LOG" 2>/dev/null && \
         grep -q "ROCm1 model buffer size" "$LOG" 2>/dev/null; then
        found=1
    fi
fi

if [ "$found" -eq 3 ]; then
    wait "$pid"
    rc=$?
    kill "$cap_mon" 2>/dev/null
    wait "$cap_mon" 2>/dev/null
    exit $rc
fi

used_mib=$(vram_mib)
if [ "$found" -eq 1 ]; then
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
else
    echo "gpu_run: no per-card report (tensor-parallel buffer?); monitor enforcing cap at ${used_mib} MiB" >&2
fi

# Load-progress check: a real stall produces no further log line. A big expert
# tensor copied from a cold compressed mmap can legitimately take minutes, so the
# window is configurable (GPU_RUN_PROGRESS_S) and the load trace counts as progress.
PROGRESS_S=${GPU_RUN_PROGRESS_S:-60}
progress=0
for _ in $(seq 1 "$PROGRESS_S"); do
    if grep -qE "llama_model_load: model loaded|build_lora_branch|llama_context: constructing|prompt eval|eval time|print_timing|scion_load:" "$LOG" 2>/dev/null; then
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
rc=$?
kill "$cap_mon" 2>/dev/null
wait "$cap_mon" 2>/dev/null
exit $rc

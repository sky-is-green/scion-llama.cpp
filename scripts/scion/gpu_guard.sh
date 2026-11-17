#!/bin/bash
# Scion GPU policy guard.
#
# Canonical two-card layout:
#   spare   = card0 / bus 07 = ROCm0 = the work horse, gets the LARGER share
#   display = card1 / bus 03 = ROCm1 = desktop, gets the SMALLER share
#
# The only accepted split is:  -ts 0.75,0.25  (spare first)
# The only accepted device order is:  --device ROCm0,ROCm1
#
# Usage:
#   gpu_guard.sh <command> [args...]
#
# It checks the command line and then execs it. If the split or device order is
# anything else, it refuses to run. This exists because a manual -ts swap put
# 75% on the display card and OOM'd the machine.

set -u

guard_argv() {
    local prev=""
    local arg val

    for arg in "$@"; do
        if [ "$prev" = "-ts" ] || [ "$prev" = "--tensor-split" ]; then
            if [ "$arg" != "0.75,0.25" ]; then
                echo "gpu_guard: refusing split '$arg'; the canonical spare/display split is 0.75,0.25 (spare gets the majority)" >&2
                return 1
            fi
            prev=""
            continue
        fi

        if [ "$prev" = "--device" ] || [ "$prev" = "-dev" ]; then
            if [ "$arg" != "ROCm0,ROCm1" ]; then
                echo "gpu_guard: refusing device order '$arg'; the canonical order is ROCm0,ROCm1 (spare first)" >&2
                return 1
            fi
            prev=""
            continue
        fi

        case "$arg" in
            --tensor-split=*)
                val="${arg#*=}"
                if [ "$val" != "0.75,0.25" ]; then
                    echo "gpu_guard: refusing split '$val'; the canonical spare/display split is 0.75,0.25" >&2
                    return 1
                fi
                ;;
            -ts|--tensor-split|--device|-dev)
                prev="$arg"
                ;;
        esac
    done

    return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    guard_argv "$@" || exit 2
    exec "$@"
fi

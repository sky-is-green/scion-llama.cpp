#!/bin/bash
# Scion GPU policy guard.
#
# Canonical two-card layout:
#   spare   = card0 / bus 07 = ROCm0 = the work horse, gets the LARGER share
#   display = card1 / bus 03 = ROCm1 = desktop, gets the SMALLER share
#
# The guard accepts any -ts split where the spare fraction is strictly greater
# than the display fraction, and only the device order ROCm0,ROCm1 (spare first).
# It refuses anything else before the process starts.

set -u

fail() {
    echo "gpu_guard: $*" >&2
    return 1
}

check_split() {
    local s=$1
    local a b
    a="${s%%,*}"
    b="${s#*,}"
    if [ "$a" = "$s" ] || [ -z "$b" ]; then
        fail "invalid split '$s'; expected spare,display (e.g. 0.75,0.25)"
        return 1
    fi
    if ! awk -v a="$a" -v b="$b" 'BEGIN { exit !(a + b > 0.99 && a + b < 1.01 && b <= 0.55) }'; then
        fail "refusing split '$s'; the display fraction must be <= 0.55 and the fractions must sum to 1"
        return 1
    fi
    return 0
}

guard_argv() {
    local prev=""
    local arg val

    for arg in "$@"; do
        if [ "$prev" = "-ts" ] || [ "$prev" = "--tensor-split" ]; then
            check_split "$arg" || return 1
            prev=""
            continue
        fi

        if [ "$prev" = "--device" ] || [ "$prev" = "-dev" ]; then
            if [ "$arg" != "ROCm0,ROCm1" ]; then
                fail "refusing device order '$arg'; the canonical order is ROCm0,ROCm1 (spare first)"
                return 1
            fi
            prev=""
            continue
        fi

        case "$arg" in
            --tensor-split=*)
                check_split "${arg#*=}" || return 1
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

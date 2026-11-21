#!/usr/bin/env bash
set -u

if (( $# == 0 )); then
    printf 'Usage: %s /path/to/qwen-tts-server [server options]\n' "$0" >&2
    exit 2
fi
if ! command -v -- "$1" >/dev/null; then
    printf 'Server executable not found: %s\n' "$1" >&2
    exit 2
fi

stopping=0
child_pid=

stop_child() {
    stopping=1
    if [[ -n $child_pid ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}

trap stop_child INT TERM

while (( ! stopping )); do
    "$@" &
    child_pid=$!
    printf '[%s] TTS server starting (PID %s)\n' "$(date -Is)" "$child_pid" >&2
    if (( stopping )); then
        stop_child
    fi
    wait "$child_pid"
    status=$?
    if (( stopping )); then
        # A trapped signal can interrupt wait before the server finishes shutting down.
        wait "$child_pid" 2>/dev/null || true
        break
    fi
    printf '[%s] TTS server PID %s exited with status %s. Restarting in 5 seconds.\n' \
        "$(date -Is)" "$child_pid" "$status" >&2
    child_pid=
    sleep 5
done

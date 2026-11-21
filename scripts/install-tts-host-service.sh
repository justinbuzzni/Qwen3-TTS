#!/usr/bin/env bash
set -euo pipefail
umask 077

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data="$HOME/.local/share/qwen3-tts-8765"
unit_dir="$HOME/.config/systemd/user"
unit=qwen3-tts-8765.service

if [[ ${1:-} == --help && $# == 1 ]]; then
    printf 'Usage: %s [--runtime-dir /path/to/worktree/.aplus]\n' "$0"
    exit 0
fi
if [[ $# == 2 && $1 == --runtime-dir ]]; then
    runtime=$2
elif (( $# == 0 )); then
    runtime="$root/.aplus"
    if [[ ! -s $runtime/qwen-tts-model.log || ! -s $runtime/qwen-tts-api-key ]]; then
        candidates=()
        for candidate in "$runtime"/worktrees/*/*/.aplus; do
            if [[ -s $candidate/qwen-tts-model.log && -s $candidate/qwen-tts-api-key ]]; then
                candidates+=("$candidate")
            fi
        done
        case ${#candidates[@]} in
            0) ;;
            1) runtime=${candidates[0]} ;;
            *)
                printf 'More than one worktree contains TTS runtime files. Select one with --runtime-dir:\n' >&2
                printf '  %s\n' "${candidates[@]}" >&2
                exit 1
                ;;
        esac
    fi
else
    printf 'Usage: %s [--runtime-dir /path/to/worktree/.aplus]\n' "$0" >&2
    exit 2
fi

if [[ ! -s $runtime/qwen-tts-model.log || ! -s $runtime/qwen-tts-api-key ]]; then
    printf 'TTS runtime files are missing in %s. Use --runtime-dir with the .aplus directory where TTS originally ran.\n' "$runtime" >&2
    exit 1
fi
model=$(tail -n 1 "$runtime/qwen-tts-model.log")
if [[ ! -s $model/model.safetensors || ! -s $model/speech_tokenizer/model.safetensors ]]; then
    printf 'The model cache recorded in %s/qwen-tts-model.log is incomplete or has moved.\n' "$runtime" >&2
    exit 1
fi
key=$(<"$runtime/qwen-tts-api-key")
if [[ ! $key =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'Expected the workspace API key in %s/qwen-tts-api-key.\n' "$runtime" >&2
    exit 1
fi

# Check host access before spending time on dependencies or copying weights.
if ! systemctl --user show-environment >/dev/null; then
    printf 'Run this script in a terminal on the host with access to the user systemd manager.\n' >&2
    exit 1
fi

mkdir -p -- "$data" "$unit_dir"
exec 9>"$data/install.lock"
flock 9

if [[ ! -d $data/model ]]; then
    staging=$(mktemp -d "$data/model.XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    cp -aL --reflink=auto -- "$model/." "$staging/"
    mv -T -- "$staging" "$data/model"
    trap - EXIT
fi

if [[ ! -f $data/venv/.ready ]]; then
    uv venv --python 3.11 "$data/venv"
    uv pip install --python "$data/venv/bin/python" \
        --index 'pytorch=https://download.pytorch.org/whl/cpu' \
        'torch==2.8.0+cpu' 'torchaudio==2.8.0+cpu' "$root[server]"
    "$data/venv/bin/qwen-tts-server" --help >/dev/null
    touch "$data/venv/.ready"
fi

printf 'QWEN_TTS_API_KEY=%s\n' "$key" > "$data/server.env.new"
mv -fT -- "$data/server.env.new" "$data/server.env"

cat > "$unit_dir/$unit.new" <<'UNIT'
[Unit]
Description=Qwen3-TTS FastAPI server on port 8765

[Service]
EnvironmentFile=%h/.local/share/qwen3-tts-8765/server.env
Environment=HF_HUB_OFFLINE=1
ExecStart=%h/.local/share/qwen3-tts-8765/venv/bin/qwen-tts-server --model %h/.local/share/qwen3-tts-8765/model --device cpu --cpu-threads 8 --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5s

[Install]
WantedBy=default.target
UNIT
mv -fT -- "$unit_dir/$unit.new" "$unit_dir/$unit"

systemctl --user daemon-reload
listener=$(ss -H -ltnp 'sport = :8765')
main_pid=$(systemctl --user show "$unit" -p MainPID --value)
if [[ -n $listener && $listener != *"pid=$main_pid,"* ]]; then
    printf 'Port 8765 is in use by another process. Stop the workspace TTS server first.\n' >&2
    exit 1
fi
systemctl --user enable "$unit"
systemctl --user restart "$unit"

for (( attempt = 0; attempt < 60; attempt++ )); do
    main_pid=$(systemctl --user show "$unit" -p MainPID --value)
    listener=$(ss -H -ltnp 'sport = :8765')
    if [[ $listener == *'0.0.0.0:8765'* && $listener == *"pid=$main_pid,"* ]] &&
        curl --noproxy '*' --fail --silent --max-time 2 http://127.0.0.1:8765/healthz >/dev/null; then
        systemctl --user show "$unit" -p ActiveState -p MainPID
        loginctl show-user "$(id -u)" -p Linger || true
        exit 0
    fi
    sleep 1
done

journalctl --user -u "$unit" -n 40 --no-pager >&2
exit 1

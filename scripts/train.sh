#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROOT"
if [[ ${1:-} == --background ]]; then
    shift
    mkdir -p outputs/launches
    NAME=$(date +%Y%m%d-%H%M%S)-$$
    nohup "$NEXARM_PYTHON" -m imitation_lab train "$@" > "outputs/launches/$NAME.log" 2>&1 < /dev/null &
    echo $! > "outputs/launches/$NAME.pid"
    echo "PID: $!"
    echo "Log: $ROOT/outputs/launches/$NAME.log"
else
    exec "$NEXARM_PYTHON" -m imitation_lab train "$@"
fi

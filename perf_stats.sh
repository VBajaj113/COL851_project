#!/usr/bin/env bash
set -euo pipefail

PID=$(pgrep -n -f forecast.py || true)
if [ -z "${PID}" ]; then
  echo "forecast.py not running; start it first." >&2
  exit 1
fi

mkdir -p metrics
# -I 1000 = 1s interval, -x , = CSV fields
perf stat -p "${PID}" -I 1000 -x , \
  -e cache-references,cache-misses,LLC-load-misses,instructions,cycles,branches,branch-misses \
  2> metrics_out/perf_raw.csv

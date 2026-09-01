#!/usr/bin/env bash
# N18 RouteC.1: one blocking command that extracts the GFN feature cache
# for train30 + calibration10 on 4 idle GPUs. No agent-level polling.
set -u

ROOT=.
cd "$ROOT"

PY="$ROOT/envs/sam3_intermot/bin/python"
OUT="$ROOT/outputs/n18/route_c"
mkdir -p "$OUT/gfn_cache" "$OUT/logs"

# fixed frozen split from N15; val25 must never be touched
mapfile -t SEQS < <("$PY" - <<'EOF'
import json
d = json.load(open("outputs/n15/n15_frozen.json"))
s = d["split"]
print("\n".join(sorted(s["train30"] + s["calibration10"])))
EOF
)

NSEQS=${#SEQS[@]}
GPUS=(0 1 2 3)
NGPU=${#GPUS[@]}
PIDS=()

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  shard=$(for ((j=i; j<NSEQS; j+=NGPU)); do echo "${SEQS[$j]}"; done | paste -sd, -)
  if [ -n "$shard" ]; then
    "$PY" scripts/build_route_c_feature_cache.py \
      --gpu "$gpu" --seqs "$shard" \
      > "$OUT/logs/cache_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "launched gpu=$gpu pid=$! seqs=$(echo "$shard" | tr ',' ' ' | wc -w)"
  fi
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "CACHE_RUNNER_FAIL"
  exit 1
fi

echo "CACHE_RUNNER_DONE nseqs=$NSEQS ngpu=$NGPU"

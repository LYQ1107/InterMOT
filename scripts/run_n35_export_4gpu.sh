#!/usr/bin/env bash
set -u

PROJECT="."
PYTHON_BIN="python"
OUTPUT_ROOT="${PROJECT}/outputs/n35/real_tape"
LOG_ROOT="${PROJECT}/outputs/n35/logs"
SEQUENCE_LIST="${PROJECT}/outputs/n34/selected_sequences.json"

mkdir -p "${LOG_ROOT}"

physical_gpus=(4 5 6 7)
sequence_list=(
    dancetrack0001 dancetrack0002 dancetrack0006 dancetrack0008
    dancetrack0012 dancetrack0015 dancetrack0016 dancetrack0020
    dancetrack0023 dancetrack0024 dancetrack0027 dancetrack0029
    dancetrack0032 dancetrack0033 dancetrack0037 dancetrack0049
    dancetrack0051 dancetrack0052 dancetrack0055 dancetrack0062
    dancetrack0066 dancetrack0068 dancetrack0069 dancetrack0072
)
pids=()

# One sequence per Python process is deliberate: the pinned official model
# can retain CUDA allocator/compile references after a session close.  A
# child process gives each sequence a fresh backend/model while keeping at
# most four SAM3 workers resident.
for shard in 0 1 2 3; do
    (
        gpu="${physical_gpus[${shard}]}"
        for sequence in "${sequence_list[@]}"; do
            index=0
            for candidate in "${sequence_list[@]}"; do
                if [[ "${candidate}" == "${sequence}" ]]; then
                    break
                fi
                index=$((index + 1))
            done
            if (( index % 4 != shard )); then
                continue
            fi
            log_path="${LOG_ROOT}/n35_export_${sequence}_gpu${gpu}.log"
            CUDA_VISIBLE_DEVICES="${gpu}" \
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
            PYTHONPATH="${PROJECT}/third_party/sam3:${PROJECT}" \
            "${PYTHON_BIN}" "${PROJECT}/scripts/run_n35_export_tape.py" \
                --sequences "${sequence}" \
                --sequence-list "${SEQUENCE_LIST}" \
                --output-root "${OUTPUT_ROOT}" \
                --gpu 0 \
                --shard 0 \
                --num-shards 1 \
                --skip-existing \
                --no-manifest >"${log_path}" 2>&1 || exit 1
        done
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

#!/usr/bin/env bash
# Export non-identifying validation examples after a managed FRAPPE run ends.
#
# This process deliberately does not touch CUDA until the training tmux session
# has exited *and* its launcher log confirms a normal FINISHED marker.  It is
# therefore safe to start while the long-running training job owns the GPU.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TRAINING_TMUX_SESSION RUN_DIRECTORY OUTPUT_PREFIX" >&2
  exit 64
fi

training_session=$1
run_dir=$(realpath "$2")
output_prefix=$(realpath -m "$3")
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
log_file="$run_dir/launcher.log"
checkpoint="$run_dir/checkpoints/last.pth.tar"
postprocess_log="$run_dir/postprocess.log"

mkdir -p "$(dirname "$output_prefix")"

while tmux has-session -t "$training_session" 2>/dev/null; do
  printf '%s waiting for training session %s\n' "$(date --iso-8601=seconds)" "$training_session" >> "$postprocess_log"
  sleep 60
done

if ! rg -q '^  FINISHED$' "$log_file"; then
  printf '%s training session ended without a FINISHED marker; no export run\n' \
    "$(date --iso-8601=seconds)" >> "$postprocess_log"
  exit 1
fi
if [[ ! -s "$checkpoint" ]]; then
  printf '%s completed run has no usable checkpoint: %s\n' \
    "$(date --iso-8601=seconds)" "$checkpoint" >> "$postprocess_log"
  exit 1
fi

export_one() {
  local suffix=$1
  shift
  pixi run python tools/export_local_reconstruction.py \
    --checkpoint "$checkpoint" \
    --dataset-root /workspace/data/frappe_rgb_800x608/imagefolder \
    --split validation \
    --channels 21 \
    --device cuda:0 \
    --output "${output_prefix}_${suffix}_reconstruction.png" \
    --comparison-output "${output_prefix}_${suffix}_comparison.png" \
    --metadata-output "${output_prefix}_${suffix}_metadata.json" \
    "$@" >> "$postprocess_log" 2>&1
}

cd "$repo_dir"
printf '%s training completed; exporting anonymous validation images\n' \
  "$(date --iso-8601=seconds)" >> "$postprocess_log"

# Keep the held-out split's aggregate quality/rate measurement alongside the
# visual outputs.  This is intentionally done only after the training tmux
# session has released CUDA.
pixi run python tools/evaluate_local_checkpoint.py \
  --checkpoint "$checkpoint" \
  --dataset-root /workspace/data/frappe_rgb_800x608/imagefolder \
  --splits test \
  --channels 21 \
  --device cuda:0 \
  --output "${output_prefix}_test_metrics.json" >> "$postprocess_log" 2>&1

# A fixed split-local index provides a prompt qualitative artifact.  The
# representative version scores the full validation split and selects the
# sample closest to its median PSNR without retaining a source filename.
export_one sample --index 0
export_one representative --representative
printf '%s post-processing completed\n' "$(date --iso-8601=seconds)" >> "$postprocess_log"

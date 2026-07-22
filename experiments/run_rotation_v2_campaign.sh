#!/usr/bin/env bash
# v2 real-rotation measurement campaign (GPU 1 only). Runs, per task:
#   - the frame=tool oracle-parity / Gate-1 ceiling cells (both rotation modes);
#   - paired exact_belief + entropy_probes slices in identity AND real mode on the
#     seen + unseen_composition splits, so v1-vs-v2 isolates the rotation axis at
#     the same schema/calibration.
# Seed 20260718, 100 eps/contract. Artifacts -> artifacts/adaptation/v2_rotation_slices.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
OUT=artifacts/adaptation/v2_rotation_slices
PY=.venv/bin/python
mkdir -p "$OUT"

for task in pick_cube push_cube; do
  echo "### parity ${task}"
  $PY experiments/run_rotation_v2_parity.py --task "$task" --output "$OUT" >/dev/null
done

for task in pick_cube push_cube; do
  for method in exact_belief entropy_probes; do
    for split in seen unseen_composition; do
      for mode in identity real; do
        echo "### slice ${task} ${method} ${split} ${mode}"
        $PY experiments/run_adaptation_slice.py \
          --task "$task" --method "$method" --split "$split" --seed 20260718 \
          --episodes 100 --num-envs 8 --rotation-mode "$mode" \
          --output "$OUT" >/dev/null
      done
    done
  done
done
echo "### DONE"

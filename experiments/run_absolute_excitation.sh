#!/usr/bin/env bash
# Absolute-excitation hold-probe evaluation (GPU 3 only). Resumable: each cell is
# hash-addressed and the runner skips a completed summary. The hold-probe schedule
# (+ the existing scale corrector downstream) attacks the all-absolute
# unseen_composition wall; Pick/seen is the delta regression control.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
PY=.venv/bin/python
OUT=artifacts/adaptation/absolute_excitation
BOUT=artifacts/adaptation/absolute_excitation_before
COMMON=(--calibration-version v2 --scale-correction --episodes 100 --num-envs 8 --output "$OUT")

# Matched BEFORE: the existing per-step pulse probe (same v2 calibration + corrector).
BEFORE=(--method factorized_grammar_probes --calibration-version v2 --scale-correction \
        --episodes 100 --num-envs 8 --seed 20260718 --output "$BOUT")
$PY experiments/run_factorized_slice.py --task push_cube --split unseen_composition "${BEFORE[@]}"
$PY experiments/run_factorized_slice.py --task pick_cube --split unseen_composition "${BEFORE[@]}"
$PY experiments/run_factorized_slice.py --task pick_cube --split seen "${BEFORE[@]}"

# PRIMARY hold-probe schedule: short holds, one round (12 probe steps of 50). Best
# operating point -- breaches the absolute cells without a delta-seen regression.
PRIMARY=(--method factorized_grammar_hold_probes --hold-steps 2 --hold-rounds 1)
for SEED in 20260718 20260719 20260720; do
  $PY experiments/run_factorized_slice.py --task push_cube --split unseen_composition --seed $SEED "${COMMON[@]}" "${PRIMARY[@]}"
  $PY experiments/run_factorized_slice.py --task pick_cube --split unseen_composition --seed $SEED "${COMMON[@]}" "${PRIMARY[@]}"
  $PY experiments/run_factorized_slice.py --task pick_cube --split seen           --seed $SEED "${COMMON[@]}" "${PRIMARY[@]}"
done

# SENSITIVITY: a longer probe (24 steps) identifies equally but eats delta control.
SENS=(--method factorized_grammar_hold_probes --hold-steps 2 --hold-rounds 2)
for SEED in 20260718 20260719 20260720; do
  $PY experiments/run_factorized_slice.py --task push_cube --split unseen_composition --seed $SEED "${COMMON[@]}" "${SENS[@]}"
  $PY experiments/run_factorized_slice.py --task pick_cube --split unseen_composition --seed $SEED "${COMMON[@]}" "${SENS[@]}"
  $PY experiments/run_factorized_slice.py --task pick_cube --split seen           --seed $SEED "${COMMON[@]}" "${SENS[@]}"
done

echo "ALL CELLS DONE"

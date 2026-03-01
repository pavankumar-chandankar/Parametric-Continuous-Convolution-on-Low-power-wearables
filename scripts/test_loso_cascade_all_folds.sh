#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

# Base config used by main_multifreq_final.py (labels + cascade params)
CONFIG="${CONFIG:-./configs/continuous_conv/wear_loso_25hz.yaml}"

# Root containing per-fold test splits:
#   data/wear/multifreq_test_loso/loso_sbj_0/test_windows_*.npz
TEST_SPLITS_ROOT="${TEST_SPLITS_ROOT:-data/wear/multifreq_test_loso}"

# Checkpoint roots (each contains loso_sbj_X/best_loso_sbj_X.pth.tar)
CKPT_ROOT_50="${CKPT_ROOT_50:-logs/deepconvlstm/50hz}"
CKPT_ROOT_25="${CKPT_ROOT_25:-logs/deepconvlstm/25hz}"
CKPT_ROOT_12="${CKPT_ROOT_12:-logs/deepconvlstm/12hz}"
CKPT_ROOT_6="${CKPT_ROOT_6:-logs/deepconvlstm/6hz}"

# Gamma-Quant flags (override via env if needed)
GAMMA_QUANT="${GAMMA_QUANT:-quantize_and_gamma}"
QUANT_BITS="${QUANT_BITS:-4}"
GAMMA_TYPE="${GAMMA_TYPE:-s_shaped}"
APPLY_GAMMA="${APPLY_GAMMA:-global}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-logs/cascade_loso_eval/${TS}}"
mkdir -p "$OUTDIR"

SUMMARY_CSV="${OUTDIR}/cascade_loso_summary.csv"

# Discover folds from checkpoint dirs (sorted)
mapfile -t FOLDS < <(ls -d "${CKPT_ROOT_25}"/loso_sbj_* 2>/dev/null | xargs -n1 basename | sort -V)

if [[ "${#FOLDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No folds found under: ${CKPT_ROOT_25}/loso_sbj_*"
  exit 1
fi

echo "============================================================"
echo "[LOSO CASCADE TEST] folds=${#FOLDS[@]}"
echo "Config:           ${CONFIG}"
echo "Test splits root: ${TEST_SPLITS_ROOT}"
echo "Checkpoints:      50=${CKPT_ROOT_50} 25=${CKPT_ROOT_25} 12=${CKPT_ROOT_12} 6=${CKPT_ROOT_6}"
echo "Output dir:       ${OUTDIR}"
echo "============================================================"

HEADER_WRITTEN=0
NUM_CLASSES=0

# tmp file to store per-fold parsed values before CSV header exists
TMP_ROWS="${OUTDIR}/_rows.tmp"
: > "$TMP_ROWS"

for fold in "${FOLDS[@]}"; do
  echo "============================================================"
  echo "[LOSO TEST] fold=${fold}"
  echo "============================================================"

  TEST_SPLIT_DIR="${TEST_SPLITS_ROOT}/${fold}"

  CKPT_50="${CKPT_ROOT_50}/${fold}/best_${fold}.pth.tar"
  CKPT_25="${CKPT_ROOT_25}/${fold}/best_${fold}.pth.tar"
  CKPT_12="${CKPT_ROOT_12}/${fold}/best_${fold}.pth.tar"
  CKPT_6="${CKPT_ROOT_6}/${fold}/best_${fold}.pth.tar"

  # Basic existence checks (fail early and clearly)
  [[ -d "$TEST_SPLIT_DIR" ]] || { echo "[ERROR] Missing test split dir: $TEST_SPLIT_DIR"; exit 1; }
  [[ -f "$CKPT_25" ]] || { echo "[ERROR] Missing ckpt: $CKPT_25"; exit 1; }
  [[ -f "$CKPT_12" ]] || { echo "[ERROR] Missing ckpt: $CKPT_12"; exit 1; }
  [[ -f "$CKPT_6"  ]] || { echo "[ERROR] Missing ckpt: $CKPT_6";  exit 1; }

  # 50 Hz is optional at runtime, but you said you trained it and want it in cascade
  [[ -f "$CKPT_50" ]] || { echo "[ERROR] Missing ckpt: $CKPT_50"; exit 1; }

  LOGFILE="${OUTDIR}/${fold}.log"

  python main_multifreq_final.py \
    --config "$CONFIG" \
    --mode test_multifreq \
    --test-split-dir "$TEST_SPLIT_DIR" \
    --ckpt-50hz "$CKPT_50" \
    --ckpt-25hz "$CKPT_25" \
    --ckpt-12hz "$CKPT_12" \
    --ckpt-6hz  "$CKPT_6" \
    --gamma_quant "$GAMMA_QUANT" \
    --quant_bits "$QUANT_BITS" \
    --gamma_type "$GAMMA_TYPE" \
    --apply_gamma "$APPLY_GAMMA" \
    2>&1 | tee "$LOGFILE"
  test "${PIPESTATUS[0]}" -eq 0

  # Parse Accuracy and Macro F1 from the log
  ACC="$(grep -E "^[[:space:]]*Accuracy:" "$LOGFILE" | tail -n 1 | awk '{print $2}' | tr -d '%')"
  MACRO_F1="$(grep -E "^[[:space:]]*Macro F1:" "$LOGFILE" | tail -n 1 | awk '{print $3}' | tr -d '%')"

  if [[ -z "$ACC" || -z "$MACRO_F1" ]]; then
    echo "[ERROR] Could not parse Accuracy/Macro F1 from log: $LOGFILE"
    exit 1
  fi

  # Parse per-class F1 lines: "Class 00: F1 =  75.45%"
  # Store in an indexed bash array by class id.
  declare -a F1S=()
  while IFS= read -r line; do
    cls="$(echo "$line" | awk '{print $2}' | tr -d ':')"
    val="$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i=="="){print $(i+1); exit}}}' | tr -d '%')"
    # remove leading zeros safely by forcing base-10
    cls_num=$((10#${cls}))
    F1S[$cls_num]="$val"
  done < <(grep -E "^[[:space:]]*Class[[:space:]]+[0-9]+:" "$LOGFILE" || true)

  if [[ "${#F1S[@]}" -eq 0 ]]; then
    echo "[ERROR] Could not parse per-class F1 lines from log: $LOGFILE"
    exit 1
  fi

  # Determine number of classes from max index seen in this fold
  MAX_IDX=0
  for i in "${!F1S[@]}"; do
    if [[ -n "${F1S[$i]:-}" && "$i" -gt "$MAX_IDX" ]]; then
      MAX_IDX="$i"
    fi
  done
  THIS_NUM_CLASSES=$((MAX_IDX + 1))

  # On first fold, write CSV header dynamically
  if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
    NUM_CLASSES="$THIS_NUM_CLASSES"
    {
      printf "fold,acc,macro_f1"
      for ((c=0; c<NUM_CLASSES; c++)); do
        printf ",class_%02d_f1" "$c"
      done
      printf "\n"
    } > "$SUMMARY_CSV"
    HEADER_WRITTEN=1
  else
    # Keep class count consistent (fail if mismatch)
    if [[ "$THIS_NUM_CLASSES" -ne "$NUM_CLASSES" ]]; then
      echo "[ERROR] Class count mismatch. First fold=$NUM_CLASSES, this fold=$THIS_NUM_CLASSES"
      exit 1
    fi
  fi

  # Write row (ensure every class has a value; if missing, write empty)
  {
    printf "%s,%s,%s" "$fold" "$ACC" "$MACRO_F1"
    for ((c=0; c<NUM_CLASSES; c++)); do
      printf ",%s" "${F1S[$c]:-}"
    done
    printf "\n"
  } >> "$SUMMARY_CSV"

  # clean per-fold array
  unset F1S
done

echo "============================================================"
echo "[DONE] Wrote per-fold summary: ${SUMMARY_CSV}"
echo "============================================================"

# Final aggregation: mean over folds (printed to terminal)
python - <<PY
import pandas as pd

path = r"${SUMMARY_CSV}"
df = pd.read_csv(path)

print("============================================================")
print("[LOSO CASCADE SUMMARY] mean over folds")
print("Folds:", len(df))
print(f"Mean Accuracy: {df['acc'].mean():.2f}%")
print(f"Mean Macro F1: {df['macro_f1'].mean():.2f}%")
print("Mean per-class F1:")
class_cols = [c for c in df.columns if c.startswith("class_") and c.endswith("_f1")]
for c in class_cols:
    print(f"  {c}: {df[c].mean():.2f}%")
print("============================================================")
PY

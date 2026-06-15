#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.001}"

python -m pip install --upgrade pip
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements-hpc.txt
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python -m src.train_ddd_classifier \
  --data-dir data/ddd \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --output models/ddd_cnn.pt \
  --metrics-output reports/ddd_metrics.json

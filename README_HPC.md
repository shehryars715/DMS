# DMS HPC Training Package

This folder is ready to copy to an HPC machine and train the DDD drowsiness CNN.

## Contents

- `src/`: project source code.
- `data/ddd/`: Driver Drowsiness Dataset with `Drowsy` and `Non Drowsy` folders.
- `requirements-hpc.txt`: minimal dependencies for CNN training.
- `hpc_train.sh`: direct Linux training launcher.
- `slurm_train_ddd.sbatch`: SLURM launcher for a GPU node.
- `models/`: output checkpoint folder.
- `reports/`: output metrics folder.

## Run With SLURM

```bash
sbatch slurm_train_ddd.sbatch
```

Optional overrides:

```bash
EPOCHS=20 BATCH_SIZE=256 sbatch slurm_train_ddd.sbatch
```

## Run Directly On A Login/Compute Node

```bash
python -m venv .venv
source .venv/bin/activate
bash hpc_train.sh
```

## Outputs

- `models/ddd_cnn.pt`: best validation checkpoint.
- `reports/ddd_metrics.json`: final test accuracy, precision, recall, F1, and confusion matrix.

## Notes

- If `torch.cuda.is_available()` prints `False` on the HPC, install the PyTorch build that matches the cluster CUDA module.
- The local CPU run was stopped after epoch 1. It had reached `val_acc=0.9842` and `val_f1=0.9854`; continue or retrain fully on the HPC for the final report.

"""Train and evaluate the DDD drowsiness CNN.

Expected data directory:
    data/ddd/Drowsy/*.jpg
    data/ddd/Non Drowsy/*.jpg

The loader is intentionally tolerant of spaces and nested folders. Any image
whose ancestor folder normalizes to "drowsy" is positive; folders that normalize
to "non drowsy", "nondrowsy", "non-drowsy", "awake", or "alert" are negative.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .ddd_model import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD, DrowsinessCNN
except ImportError:  # pragma: no cover - allows `python src/train_ddd_classifier.py`
    from ddd_model import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD, DrowsinessCNN


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSITIVE_CLASS = "Drowsy"
NEGATIVE_CLASS = "Non Drowsy"
CLASS_TO_INDEX = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output: Path
    metrics_output: Path
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    seed: int
    num_workers: int
    val_size: float
    test_size: float
    threshold: float
    init_checkpoint: Path | None


def normalize_label_name(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def infer_label_from_path(path: Path, data_dir: Path) -> str | None:
    negative_names = {"non drowsy", "nondrowsy", "not drowsy", "awake", "alert", "normal"}
    positive_names = {"drowsy", "sleepy", "closed"}
    for part in path.relative_to(data_dir).parts[:-1]:
        normalized = normalize_label_name(part)
        if normalized in negative_names:
            return NEGATIVE_CLASS
        if normalized in positive_names:
            return POSITIVE_CLASS
    return None


def discover_images(data_dir: Path) -> tuple[list[Path], list[int]]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DDD dataset path does not exist: {data_dir}. "
            "Download Kaggle DDD and place it under ./data/ddd/."
        )

    paths: list[Path] = []
    labels: list[int] = []
    skipped = 0

    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_name = infer_label_from_path(path, data_dir)
        if label_name is None:
            skipped += 1
            continue
        paths.append(path)
        labels.append(CLASS_TO_INDEX[label_name])

    if not paths:
        raise RuntimeError(
            f"No labeled images found in {data_dir}. Expected class folders named "
            "'Drowsy' and 'Non Drowsy' or equivalent labels."
        )

    counts = {NEGATIVE_CLASS: labels.count(0), POSITIVE_CLASS: labels.count(1)}
    if min(counts.values()) == 0:
        raise RuntimeError(f"Both classes are required. Found counts: {counts}")

    if skipped:
        print(f"Skipped {skipped} images because no DDD class folder could be inferred.")
    print(f"Discovered {len(paths)} images: {counts}")
    return paths, labels


def stratified_split(
    paths: list[Path],
    labels: list[int],
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    train_size = 1.0 - val_size - test_size
    if train_size <= 0:
        raise ValueError("val_size + test_size must be less than 1.0")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


class DDDDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        img_size: int,
        augment: bool,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        self.mean = np.asarray(DEFAULT_MEAN, dtype=np.float32)
        self.std = np.asarray(DEFAULT_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            image = self._augment(image)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        label = np.float32(self.labels[index])
        return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        h, w = image.shape[:2]
        if random.random() < 0.75:
            angle = random.uniform(-10.0, 10.0)
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if random.random() < 0.75:
            factor = random.uniform(0.75, 1.25)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return image


def make_loader(
    paths: list[Path],
    labels: list[int],
    config: TrainConfig,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = DDDDataset(paths, labels, config.img_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, np.asarray(all_probs), np.asarray(all_labels)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "classes": [NEGATIVE_CLASS, POSITIVE_CLASS],
        "threshold": threshold,
    }


def train(config: TrainConfig) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    paths, labels = discover_images(config.data_dir)
    split = stratified_split(paths, labels, config.val_size, config.test_size, config.seed)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split

    train_loader = make_loader(train_paths, train_labels, config, augment=True, shuffle=True)
    val_loader = make_loader(val_paths, val_labels, config, augment=False, shuffle=False)
    test_loader = make_loader(test_paths, test_labels, config, augment=False, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    if config.init_checkpoint is not None:
        if not config.init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {config.init_checkpoint}")
        checkpoint = torch.load(config.init_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
        if config.init_checkpoint.resolve() != config.output.resolve():
            shutil.copy2(config.init_checkpoint, config.output)
        print(
            f"Warm-started from {config.init_checkpoint} "
            f"(starting best_val_f1={best_val_f1:.4f})"
        )

    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss, train_probs, train_y = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_probs, val_y = run_epoch(model, val_loader, criterion, device)
        train_metrics = compute_metrics(train_y, train_probs, config.threshold)
        val_metrics = compute_metrics(val_y, val_probs, config.threshold)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_f1": float(train_metrics["f1"]),
            "val_f1": float(val_metrics["f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(row)
        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": config.img_size,
                    "threshold": config.threshold,
                    "dropout": 0.25,
                    "classes": [NEGATIVE_CLASS, POSITIVE_CLASS],
                    "config": {**asdict(config), "data_dir": str(config.data_dir), "output": str(config.output), "metrics_output": str(config.metrics_output)},
                    "best_val_f1": best_val_f1,
                },
                config.output,
            )
            print(f"Saved best checkpoint to {config.output}")

    checkpoint = torch.load(config.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_probs, test_y = run_epoch(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_y, test_probs, config.threshold)
    results = {
        "test_loss": float(test_loss),
        "test": test_metrics,
        "best_val_f1": best_val_f1,
        "history": history,
        "split_sizes": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
    }

    config.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Saved metrics to {config.metrics_output}")
    return results


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ddd"))
    parser.add_argument("--output", type=Path, default=Path("models/ddd_cnn.pt"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/ddd_metrics.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from before training.",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

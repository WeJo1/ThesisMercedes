from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MildGaussianNoise:
    """Add mild gaussian noise after conversion to tensor."""

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.015)) -> None:
        self.std_min, self.std_max = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.std_min, self.std_max)
        if std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomJPEGCompression:
    """Apply random jpeg compression artifacts to PIL images."""

    def __init__(self, quality_range: Tuple[int, int] = (88, 98), p: float = 0.5) -> None:
        self.quality_min, self.quality_max = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        quality = random.randint(self.quality_min, self.quality_max)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        return compressed.convert("RGB")


@dataclass(frozen=True)
class PairSample:
    original_path: Path
    generated_path: Path
    deviation_percent: float


class PairRegressionDataset(Dataset):
    """Regression dataset for product pair quality estimation."""

    def __init__(
        self,
        samples: Sequence[PairSample],
        train: bool,
    ) -> None:
        self.samples = list(samples)
        self.train = train
        self.transform = self._build_transform(train=train)

    def _build_transform(self, train: bool) -> transforms.Compose:
        pipeline: List[transforms.Compose] = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        if train:
            pipeline.extend(
                [
                    transforms.ColorJitter(brightness=0.06, contrast=0.06),
                    RandomJPEGCompression(quality_range=(88, 98), p=0.5),
                ]
            )

        pipeline.append(transforms.ToTensor())

        if train:
            pipeline.append(MildGaussianNoise(std_range=(0.0, 0.015)))

        pipeline.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return transforms.Compose(pipeline)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        img_1 = Image.open(sample.original_path).convert("RGB")
        img_2 = Image.open(sample.generated_path).convert("RGB")

        x1 = self.transform(img_1)
        x2 = self.transform(img_2)

        label = torch.tensor(sample.deviation_percent / 100.0, dtype=torch.float32)

        return {
            "x1": x1,
            "x2": x2,
            "y": label,
            "meta": {
                "orig": str(sample.original_path),
                "gen": str(sample.generated_path),
                "deviation_percent": sample.deviation_percent,
            },
        }


def _read_pair_rows(csv_path: Path) -> List[Dict[str, str]]:
    required_columns = {"original_path", "generated_path", "deviation_percent"}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = required_columns - fieldnames
        if missing_columns:
            raise ValueError(
                f"CSV '{csv_path}' misses required columns: {sorted(missing_columns)}"
            )

        return list(reader)


def _parse_sample(row: Dict[str, str], row_idx: int, data_root: Path) -> PairSample:
    original_rel = (row.get("original_path") or "").strip()
    generated_rel = (row.get("generated_path") or "").strip()
    deviation_raw = row.get("deviation_percent")

    if not original_rel:
        raise ValueError(f"Row {row_idx}: 'original_path' is empty")
    if not generated_rel:
        raise ValueError(f"Row {row_idx}: 'generated_path' is empty")

    original_path = (data_root / original_rel).resolve()
    generated_path = (data_root / generated_rel).resolve()
    data_root_resolved = data_root.resolve()

    if data_root_resolved not in original_path.parents and original_path != data_root_resolved:
        raise ValueError(
            f"Row {row_idx}: 'original_path' points outside data root: {original_rel}"
        )
    if data_root_resolved not in generated_path.parents and generated_path != data_root_resolved:
        raise ValueError(
            f"Row {row_idx}: 'generated_path' points outside data root: {generated_rel}"
        )

    if not original_path.exists():
        raise FileNotFoundError(
            f"Row {row_idx}: original image does not exist: '{original_path}'"
        )
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Row {row_idx}: generated image does not exist: '{generated_path}'"
        )

    try:
        deviation_value = float(deviation_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Row {row_idx}: invalid 'deviation_percent' value: {deviation_raw!r}"
        ) from error

    if math.isnan(deviation_value):
        raise ValueError(f"Row {row_idx}: 'deviation_percent' is NaN")
    if not 0.0 <= deviation_value <= 100.0:
        raise ValueError(
            f"Row {row_idx}: 'deviation_percent'={deviation_value} out of range [0, 100]"
        )

    return PairSample(
        original_path=original_path,
        generated_path=generated_path,
        deviation_percent=deviation_value,
    )


def load_samples_from_csv(csv_path: Path, data_root: Path) -> List[PairSample]:
    rows = _read_pair_rows(csv_path=csv_path)
    if not rows:
        raise ValueError(f"CSV '{csv_path}' is empty")

    samples = [
        _parse_sample(row=row, row_idx=row_idx, data_root=data_root)
        for row_idx, row in enumerate(rows, start=2)
    ]
    return samples


def split_samples(
    samples: Sequence[PairSample],
    train_ratio: float,
    seed: int,
) -> Tuple[List[PairSample], List[PairSample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to perform train/val split")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_idx = int(round(len(indices) * train_ratio))
    split_idx = max(1, min(split_idx, len(indices) - 1))

    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]
    return train_samples, val_samples


def create_pair_datasets(config: Dict[str, object]) -> Tuple[PairRegressionDataset, PairRegressionDataset]:
    """Create train and validation datasets.

    Supports two modes:
    1) Single CSV + split with ratio/seed.
    2) Separate train_csv and val_csv.
    """

    data_root = Path(str(config.get("data_root", "data")))
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: '{data_root}'")

    train_csv = config.get("train_csv")
    val_csv = config.get("val_csv")

    if train_csv and val_csv:
        train_samples = load_samples_from_csv(Path(str(train_csv)), data_root=data_root)
        val_samples = load_samples_from_csv(Path(str(val_csv)), data_root=data_root)
    else:
        csv_path = Path(str(config.get("pairs_csv", data_root / "pairs.csv")))
        samples = load_samples_from_csv(csv_path=csv_path, data_root=data_root)
        ratio = float(config.get("train_ratio", 0.8))
        seed = int(config.get("seed", 42))
        train_samples, val_samples = split_samples(samples=samples, train_ratio=ratio, seed=seed)

    train_dataset = PairRegressionDataset(samples=train_samples, train=True)
    val_dataset = PairRegressionDataset(samples=val_samples, train=False)
    return train_dataset, val_dataset

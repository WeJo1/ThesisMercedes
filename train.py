from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import create_pair_datasets
from model import SiamesePairRegressor, print_model_summary


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae_percent: float
    val_rmse_percent: float
    val_spearman: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Siamese pair regression model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")

    required_top_keys = {"run_name", "data", "train", "model", "amp"}
    missing = required_top_keys - set(config.keys())
    if missing:
        raise ValueError(f"Config misses required top-level keys: {sorted(missing)}")

    for section in ("data", "train", "model"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config section '{section}' must be a mapping")

    required_data_keys = {"csv_path", "data_root", "val_split", "seed"}
    missing_data = required_data_keys - set(config["data"].keys())
    if missing_data:
        raise ValueError(f"Config section 'data' misses keys: {sorted(missing_data)}")

    required_train_keys = {"batch_size", "epochs", "lr", "weight_decay", "num_workers"}
    missing_train = required_train_keys - set(config["train"].keys())
    if missing_train:
        raise ValueError(f"Config section 'train' misses keys: {sorted(missing_train)}")

    return config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_run_dir(run_name: str) -> Path:
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(config_path: Path, run_dir: Path) -> None:
    destination = run_dir / "config.yaml"
    shutil.copyfile(config_path, destination)


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["train"]

    val_split = float(data_cfg["val_split"])
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"data.val_split must be in (0, 1), got {val_split}")

    dataset_config = {
        "data_root": data_cfg["data_root"],
        "pairs_csv": data_cfg["csv_path"],
        "train_ratio": 1.0 - val_split,
        "seed": int(data_cfg["seed"]),
    }

    train_dataset, val_dataset = create_pair_datasets(dataset_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Spearman input lengths must match")
    if len(x) < 2:
        return 0.0

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    x_ranks = rankdata(x_arr)
    y_ranks = rankdata(y_arr)

    x_std = x_ranks.std()
    y_std = y_ranks.std()
    if x_std == 0.0 or y_std == 0.0:
        return 0.0

    corr_matrix = np.corrcoef(x_ranks, y_ranks)
    corr = float(corr_matrix[0, 1])
    if math.isnan(corr):
        return 0.0
    return corr


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float, float]:
    model.eval()
    val_losses: List[float] = []
    all_preds_percent: List[float] = []
    all_targets_percent: List[float] = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch in loader:
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(x1, x2)
                pred = outputs["score"]
                loss = criterion(pred, y)

            val_losses.append(float(loss.item()))

            pred_percent = (pred * 100.0).detach().cpu().numpy()
            target_percent = (y * 100.0).detach().cpu().numpy()
            all_preds_percent.extend(pred_percent.tolist())
            all_targets_percent.extend(target_percent.tolist())

    preds_np = np.asarray(all_preds_percent, dtype=np.float64)
    targets_np = np.asarray(all_targets_percent, dtype=np.float64)

    mae_percent = float(np.mean(np.abs(preds_np - targets_np)))
    rmse_percent = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    spearman = spearman_correlation(all_preds_percent, all_targets_percent)
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0

    return val_loss, mae_percent, rmse_percent, spearman


def append_metrics(metrics_path: Path, metrics: EpochMetrics) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mae_percent",
        "val_rmse_percent",
        "val_spearman",
        "lr",
    ]
    write_header = not metrics_path.exists()

    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": metrics.epoch,
                "train_loss": f"{metrics.train_loss:.8f}",
                "val_loss": f"{metrics.val_loss:.8f}",
                "val_mae_percent": f"{metrics.val_mae_percent:.6f}",
                "val_rmse_percent": f"{metrics.val_rmse_percent:.6f}",
                "val_spearman": f"{metrics.val_spearman:.6f}",
                "lr": f"{metrics.lr:.10f}",
            }
        )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_mae: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
        },
        checkpoint_path,
    )


def train() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    deterministic = bool(data_cfg.get("deterministic", True))
    seed_everything(int(data_cfg["seed"]), deterministic=deterministic)

    run_dir = make_run_dir(str(config["run_name"]))
    save_config_snapshot(config_path, run_dir)
    metrics_path = run_dir / "metrics.csv"
    best_ckpt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(config)

    model = SiamesePairRegressor(
        freeze_backbone_epochs=int(model_cfg.get("freeze_backbone_epochs", 0))
    ).to(device)
    print_model_summary(model)

    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    use_amp = bool(config["amp"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")

    num_epochs = int(train_cfg["epochs"])
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience: Optional[int] = None

    if "early_stopping" in config and config["early_stopping"] is not None:
        early_cfg = config["early_stopping"]
        if not isinstance(early_cfg, dict):
            raise ValueError("Config section 'early_stopping' must be a mapping")
        if "patience" in early_cfg and early_cfg["patience"] is not None:
            early_stopping_patience = int(early_cfg["patience"])

    for epoch in range(1, num_epochs + 1):
        model.set_backbone_trainable(epoch=epoch - 1)
        model.train()

        train_losses: List[float] = []
        autocast_device = "cuda" if device.type == "cuda" else "cpu"

        for batch in train_loader:
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(x1, x2)
                pred = outputs["score"]
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss, val_mae_percent, val_rmse_percent, val_spearman = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(val_mae_percent)
        current_lr = float(optimizer.param_groups[0]["lr"])

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mae_percent=val_mae_percent,
            val_rmse_percent=val_rmse_percent,
            val_spearman=val_spearman,
            lr=current_lr,
        )
        append_metrics(metrics_path, metrics)

        save_checkpoint(
            checkpoint_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_mae=best_val_mae,
        )

        improved = val_mae_percent < best_val_mae
        if improved:
            best_val_mae = val_mae_percent
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_mae=best_val_mae,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_MAE%={val_mae_percent:.3f} | "
            f"val_RMSE%={val_rmse_percent:.3f} | "
            f"val_Spearman={val_spearman:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                "Stop early because validation MAE did not improve for "
                f"{early_stopping_patience} epoch(s)."
            )
            break


if __name__ == "__main__":
    train()

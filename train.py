import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.pair_dataset import PairDataset
from src.models.pair_scorer import PairScorer
from src.train_utils import evaluate, train_one_epoch
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import CSVLogger, log
from src.utils.seed import set_seed


def build_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train a pair scoring model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or cfg["run_name"]
    run_dir = Path("runs") / run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_csv = run_dir / "metrics.csv"

    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = build_device(cfg.get("device", "auto"))
    log(f"Use device: {device}")

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
    ])

    train_ds = PairDataset(cfg["data"]["train_csv"], transform=transform)
    val_ds = PairDataset(cfg["data"]["val_csv"], transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = PairScorer(**cfg["model"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    csv_logger = CSVLogger(metrics_csv)
    best_val_loss = float("inf")

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        log(f"Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_every=cfg["train"].get("log_every", 10),
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        log(
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        csv_logger.log_row(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
        )

        if epoch % cfg["checkpoint"]["save_every"] == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pt",
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                },
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                },
            )

    log(f"Finished. Saved outputs in: {run_dir}")


if __name__ == "__main__":
    main()

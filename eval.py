import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.pair_dataset import PairDataset
from src.models.pair_scorer import PairScorer
from src.train_utils import evaluate
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import log
from src.utils.seed import set_seed


def build_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on test pairs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = build_device(cfg.get("device", "auto"))

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
    ])
    test_ds = PairDataset(cfg["data"]["test_csv"], transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = PairScorer(**cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    log(f"test_loss={metrics['loss']:.4f} test_acc={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()

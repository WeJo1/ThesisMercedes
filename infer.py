import argparse

import torch
from PIL import Image
from torchvision import transforms

from src.models.pair_scorer import PairScorer
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def build_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def main():
    parser = argparse.ArgumentParser(description="Run inference on one image pair.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-a", type=str, required=True)
    parser.add_argument("--image-b", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = build_device(cfg.get("device", "auto"))

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
    ])

    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")

    tensor_a = transform(image_a).unsqueeze(0).to(device)
    tensor_b = transform(image_b).unsqueeze(0).to(device)

    model = PairScorer(**cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logit = model(tensor_a, tensor_b)
        score = torch.sigmoid(logit).item()

    print(f"score={score:.6f}")


if __name__ == "__main__":
    main()

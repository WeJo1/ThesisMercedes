import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.pair_scorer import PairScorer
from src.utils.checkpoint import load_checkpoint


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer deviation for one image pair.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--original_img", type=str, required=True)
    parser.add_argument("--generated_img", type=str, required=True)
    args = parser.parse_args()

    device = build_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get("config", {}).get("model", {})

    model = PairScorer(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transform()
    original_img = Image.open(Path(args.original_img)).convert("RGB")
    generated_img = Image.open(Path(args.generated_img)).convert("RGB")

    input_a = transform(original_img).unsqueeze(0).to(device)
    input_b = transform(generated_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_deviation = float(model(input_a, input_b).item())

    pred_score_0_1 = max(0.0, min(1.0, 1.0 - (pred_deviation / 100.0)))
    output = {
        "pred_deviation_percent": pred_deviation,
        "pred_score_0_1": pred_score_0_1,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

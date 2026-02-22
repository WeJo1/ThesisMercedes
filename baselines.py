import argparse
from pathlib import Path

import lpips
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms


IMAGE_SIZE = 224


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return data_root / path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column. Tried: {candidates}")


def clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline metrics for image pairs.")
    parser.add_argument("--csv", type=str, required=True, help="Path to pairs.csv")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_csv = Path(args.output_csv) if args.output_csv else csv_path.parent / "baseline_predictions.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device)
    clip_model.eval()

    pil_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_tensor = transforms.ToTensor()

    df = pd.read_csv(csv_path)
    original_col = find_column(df, ["original_path", "image_a"])
    generated_col = find_column(df, ["generated_path", "image_b"])
    target_col = find_column(df, ["target_deviation_percent", "label"])

    data_root = Path(args.data_root)
    rows = []

    with torch.no_grad():
        for row in df.itertuples(index=False):
            original_path = getattr(row, original_col)
            generated_path = getattr(row, generated_col)
            target_value = float(getattr(row, target_col))

            image_a = Image.open(resolve_image_path(original_path, data_root)).convert("RGB")
            image_b = Image.open(resolve_image_path(generated_path, data_root)).convert("RGB")

            image_a = pil_resize(image_a)
            image_b = pil_resize(image_b)

            np_a = np.asarray(image_a).astype(np.float32) / 255.0
            np_b = np.asarray(image_b).astype(np.float32) / 255.0
            ssim_similarity = float(structural_similarity(np_a, np_b, channel_axis=-1, data_range=1.0))

            lpips_a = (to_tensor(image_a).unsqueeze(0).to(device) * 2.0) - 1.0
            lpips_b = (to_tensor(image_b).unsqueeze(0).to(device) * 2.0) - 1.0
            lpips_distance = float(lpips_model(lpips_a, lpips_b).item())

            clip_a = clip_preprocess(image_a).unsqueeze(0).to(device)
            clip_b = clip_preprocess(image_b).unsqueeze(0).to(device)
            feat_a = clip_model.encode_image(clip_a)
            feat_b = clip_model.encode_image(clip_b)
            feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            clip_similarity = float((feat_a * feat_b).sum(dim=-1).item())

            ssim_pred = clamp_percent((1.0 - ssim_similarity) * 100.0)
            lpips_pred = clamp_percent(max(0.0, min(1.0, lpips_distance)) * 100.0)
            clip_pred = clamp_percent((1.0 - clip_similarity) * 100.0)

            rows.append(
                {
                    "original_path": original_path,
                    "generated_path": generated_path,
                    "target_deviation_percent": target_value,
                    "ssim_similarity": ssim_similarity,
                    "lpips_distance": lpips_distance,
                    "clip_similarity": clip_similarity,
                    "ssim_pred_deviation_percent": ssim_pred,
                    "lpips_pred_deviation_percent": lpips_pred,
                    "clip_pred_deviation_percent": clip_pred,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved baseline predictions: {output_csv}")


if __name__ == "__main__":
    main()

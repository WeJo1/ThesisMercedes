import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.models.pair_scorer import PairScorer
from src.utils.checkpoint import load_checkpoint


def build_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return data_root / image_path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column. Tried: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pair regression checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True, help="Path to pairs.csv")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = resolve_run_dir(checkpoint_path)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = build_device()
    ckpt = load_checkpoint(str(checkpoint_path), map_location=device)
    cfg = ckpt.get("config", {})

    image_size = cfg.get("data", {}).get("image_size", 224)
    model_cfg = cfg.get("model", {})

    model = PairScorer(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    df = pd.read_csv(args.csv)
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

            tensor_a = transform(image_a).unsqueeze(0).to(device)
            tensor_b = transform(image_b).unsqueeze(0).to(device)

            pred_value = model(tensor_a, tensor_b).item()
            abs_error = abs(pred_value - target_value)

            rows.append(
                {
                    "original_path": original_path,
                    "generated_path": generated_path,
                    "target_deviation_percent": target_value,
                    "pred_deviation_percent": pred_value,
                    "abs_error_percent": abs_error,
                }
            )

    results_df = pd.DataFrame(rows)
    mae = float(results_df["abs_error_percent"].mean())
    rmse = float(((results_df["pred_deviation_percent"] - results_df["target_deviation_percent"]) ** 2).mean() ** 0.5)
    spearman = float(results_df["pred_deviation_percent"].corr(results_df["target_deviation_percent"], method="spearman"))

    report = {
        "mae_percent": mae,
        "rmse_percent": rmse,
        "spearman_correlation": spearman,
        "num_samples": int(len(results_df)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    top_errors = results_df.sort_values("abs_error_percent", ascending=False).head(50)
    top_errors.to_csv(output_dir / "top_errors.csv", index=False)

    print(f"MAE (%): {mae:.4f}")
    print(f"RMSE (%): {rmse:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Samples: {len(results_df)}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

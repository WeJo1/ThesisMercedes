import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASELINES = {
    "ssim": "ssim_pred_deviation_percent",
    "lpips": "lpips_pred_deviation_percent",
    "clip": "clip_pred_deviation_percent",
}


def compute_metrics(df: pd.DataFrame, pred_col: str, target_col: str) -> dict:
    errors = df[pred_col] - df[target_col]
    abs_errors = errors.abs()
    mae = float(abs_errors.mean())
    rmse = float((errors.pow(2).mean()) ** 0.5)
    spearman = float(df[pred_col].corr(df[target_col], method="spearman"))
    return {
        "mae_percent": mae,
        "rmse_percent": rmse,
        "spearman_correlation": spearman,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions.")
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    target_col = "target_deviation_percent"

    report = {
        "num_samples": int(len(df)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baselines": {},
    }

    for name, pred_col in BASELINES.items():
        if pred_col not in df.columns:
            raise ValueError(f"Missing prediction column: {pred_col}")
        metrics = compute_metrics(df, pred_col, target_col)
        report["baselines"][name] = metrics
        print(f"{name.upper()} -> MAE: {metrics['mae_percent']:.4f}, RMSE: {metrics['rmse_percent']:.4f}, Spearman: {metrics['spearman_correlation']:.4f}")

    report_path = output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved baseline report: {report_path}")


if __name__ == "__main__":
    main()

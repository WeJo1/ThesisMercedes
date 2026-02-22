from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairRegressionDataset, load_samples_from_csv
from model import SiamesePairRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pair regression checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--csv", type=str, required=True, help="Path to pairs.csv")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: checkpoint parent directory)",
    )
    return parser.parse_args()


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


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        raise ValueError("Spearman input lengths must match")
    if len(x) < 2:
        return 0.0

    x_ranks = rankdata(x)
    y_ranks = rankdata(y)

    x_std = x_ranks.std()
    y_std = y_ranks.std()
    if x_std == 0.0 or y_std == 0.0:
        return 0.0

    corr = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def save_top_errors(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "original_path",
        "generated_path",
        "target_deviation_percent",
        "pred_deviation_percent",
        "abs_error_percent",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    csv_path = Path(args.csv)
    data_root = Path(args.data_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples_from_csv(csv_path=csv_path, data_root=data_root)
    dataset = PairRegressionDataset(samples=samples, train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = SiamesePairRegressor()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preds_percent: List[float] = []
    targets_percent: List[float] = []
    error_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            outputs = model(x1, x2)
            pred = outputs["deviation_percent"].detach().cpu().numpy()
            target = (y * 100.0).detach().cpu().numpy()

            preds_percent.extend(pred.tolist())
            targets_percent.extend(target.tolist())

            meta_batch = batch["meta"]
            orig_list = meta_batch["orig"]
            gen_list = meta_batch["gen"]

            for i in range(len(pred)):
                abs_error = abs(float(pred[i]) - float(target[i]))
                error_rows.append(
                    {
                        "original_path": orig_list[i],
                        "generated_path": gen_list[i],
                        "target_deviation_percent": f"{float(target[i]):.6f}",
                        "pred_deviation_percent": f"{float(pred[i]):.6f}",
                        "abs_error_percent": f"{abs_error:.6f}",
                    }
                )

    preds_np = np.asarray(preds_percent, dtype=np.float64)
    targets_np = np.asarray(targets_percent, dtype=np.float64)

    mae = float(np.mean(np.abs(preds_np - targets_np)))
    rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    spearman = spearman_correlation(preds_np, targets_np)

    report = {
        "mae_percent": mae,
        "rmse_percent": rmse,
        "spearman": spearman,
        "num_samples": int(len(targets_np)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    top_errors = sorted(error_rows, key=lambda row: float(row["abs_error_percent"]), reverse=True)[:50]
    save_top_errors(output_dir / "top_errors.csv", top_errors)

    print(f"Samples: {report['num_samples']}")
    print(f"MAE (%): {report['mae_percent']:.6f}")
    print(f"RMSE (%): {report['rmse_percent']:.6f}")
    print(f"Spearman: {report['spearman']:.6f}")
    print(f"Saved report: {report_path}")
    print(f"Saved top errors: {output_dir / 'top_errors.csv'}")


if __name__ == "__main__":
    main()

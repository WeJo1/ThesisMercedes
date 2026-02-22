# Pair Comparison Research Template

Minimal Python 3.10+ codebase to train and evaluate a model on image pairs (original vs generated).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

CSV files must contain image paths and labels, for example:

```text
image_a,image_b,label
path/to/original.png,path/to/generated.png,12.5
```

## Core commands

### 1) Training

```bash
python train.py --config configs/default.yaml --run-name exp1
```

### 2) Evaluation

```bash
python eval.py \
  --checkpoint runs/exp1/checkpoints/best.pt \
  --csv data/test_pairs.csv \
  --data_root .
```

### 3) Inference

```bash
python infer.py \
  --checkpoint runs/exp1/checkpoints/best.pt \
  --original_img path/to/original.png \
  --generated_img path/to/generated.png
```

## Baseline similarity metrics

Use these scripts to compute SSIM, LPIPS (AlexNet), and CLIP cosine similarity baselines:

```bash
python baselines.py --csv data/test_pairs.csv --data_root .
python baseline_eval.py --predictions_csv data/baseline_predictions.csv
```

Additional dependencies used for baselines are listed in `requirements.txt`: `scikit-image`, `lpips`, and `open_clip_torch`.

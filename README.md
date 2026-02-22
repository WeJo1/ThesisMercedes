# Pair Comparison Research Template

Minimal Python 3.10+ codebase to train and evaluate a model on image pairs (original vs generated).

## Structure

- `train.py`: training entrypoint
- `eval.py`: evaluation entrypoint
- `infer.py`: inference for one image pair
- `configs/default.yaml`: config file
- `src/`: model, data loading, utilities
- `runs/<run_name>/`: checkpoints and CSV metrics

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

CSV files in `data/` must contain:

```text
image_a,image_b,label
path/to/original.png,path/to/generated.png,1
```

## Train

```bash
python train.py --config configs/default.yaml --run-name exp1
```

## Evaluate

```bash
python eval.py --config configs/default.yaml --checkpoint runs/exp1/checkpoints/best.pt
```

## Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint runs/exp1/checkpoints/best.pt \
  --image-a path/to/original.png \
  --image-b path/to/generated.png
```

from typing import Dict

import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, log_every: int = 10) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for step, (image_a, image_b, labels) in enumerate(loader, start=1):
        image_a = image_a.to(device)
        image_b = image_b.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(image_a, image_b)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits.detach(), labels)

        if step % log_every == 0:
            print(f"  step={step:04d} loss={loss.item():.4f}")

    n = len(loader)
    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for image_a, image_b, labels in loader:
        image_a = image_a.to(device)
        image_b = image_b.to(device)
        labels = labels.to(device)

        logits = model(image_a, image_b)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)

    n = len(loader)
    return {"loss": total_loss / n, "acc": total_acc / n}

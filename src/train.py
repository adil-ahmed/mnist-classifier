from typing import Dict
import torch
from torch import nn, optim

def get_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, device="cpu") -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)          # forward
        loss = criterion(logits, labels)

        optimizer.zero_grad()           # clear old grads
        loss.backward()                 # backward (compute grads)
        optimizer.step()                # update weights

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device="cpu") -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": 100.0 * correct / total}


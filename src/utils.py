import os
import random
import logging
import torch
import numpy as np


def set_seed(seed=2025):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(name):
    """Initialize and return a logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def save_checkpoint(model, optimizer, epoch, save_dir, prefix='ckpt'):
    """Save model and optimizer state."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load states from checkpoint."""
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', None)


def evaluate_accuracy(model, loader, device):
    """Compute classification accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

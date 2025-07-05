import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.config import CONFIG
from src.data.dataset import get_dataloader
from src.models.vit import ViT3D
from src.models.adapters import ParallelAdapter
from src.models.heads import ClassificationHead
from src.utils import set_seed, init_logger, save_checkpoint, evaluate_accuracy

def main(task='driver'):
    cfg = CONFIG['finetune']
    paths = CONFIG['paths']

    set_seed()
    logger = init_logger(f'finetune_{task}')
    writer = SummaryWriter(log_dir=os.path.join(paths['save_dir'], 'logs', f'finetune_{task}'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(paths['finetune_data'], batch_size=cfg['batch_size'])

    backbone = ViT3D()
    adapter = ParallelAdapter(
        embed_dim=backbone.patch_embed.proj.out_channels,
        adapter_rank=cfg['adapter_rank']
    )
    num_classes = 20 if task == 'driver' else 2
    head = ClassificationHead(
        embed_dim=backbone.patch_embed.proj.out_channels,
        num_classes=num_classes
    )
    model = nn.Sequential(backbone, adapter, head).to(device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg['epochs']):
        model.train()
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            logits = model(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, train_loader, device)
        logger.info(f"Epoch {epoch} Accuracy {acc:.4f}")
        writer.add_scalar('Accuracy/val', acc, epoch)
        save_checkpoint(model, optimizer, epoch, paths['save_dir'], prefix=f'finetune_{task}')

    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['driver', 'status'], default='driver')
    args = parser.parse_args()
    main(task=args.task)

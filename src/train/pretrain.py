import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.config import CONFIG
from src.data.dataset import get_dataloader
from src.models.vit import ViT3D
from src.models.heads import ContrastiveHead
from src.utils import set_seed, init_logger, save_checkpoint

def main():
    cfg = CONFIG['pretrain']
    paths = CONFIG['paths']

    set_seed(42)
    logger = init_logger('pretrain')
    writer = SummaryWriter(log_dir=os.path.join(paths['save_dir'], 'logs', 'pretrain'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(paths['pretrain_data'], batch_size=cfg['batch_size'])

    backbone = ViT3D()
    head = ContrastiveHead(embed_dim=backbone.patch_embed.proj.out_channels)
    model = nn.Sequential(backbone, head).to(device)

    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(cfg['epochs']):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            z1 = model(batch)
            z2 = model(batch)
            logits = torch.matmul(z1, z2.t()) / 0.5
            labels = torch.arange(logits.size(0), device=device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        save_checkpoint(model, optimizer, epoch, paths['save_dir'], prefix='pretrain')

    writer.close()

if __name__ == '__main__':
    main()

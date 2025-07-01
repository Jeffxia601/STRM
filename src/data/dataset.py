import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SpatioTemporalDataset(Dataset):
    """Dataset for spatio-temporal data: (time_steps × spatial_patches × features)."""
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted([os.path.join(data_dir, f)
                             for f in os.listdir(data_dir)
                             if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # shape: (T, H, W, C)
        if self.transform:
            data = self.transform(data)
        return torch.from_numpy(data).float()


def get_dataloader(data_dir, batch_size, shuffle=True, num_workers=4):
    """Construct DataLoader for training or evaluation."""
    dataset = SpatioTemporalDataset(data_dir)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

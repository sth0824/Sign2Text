# app/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, data_dir: str, label2idx: dict):
        self.samples = []
        self.label2idx = label2idx

        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                label = file.replace(".npy", "")
                label_idx = self.label2idx.get(label)
                if label_idx is not None:
                    path = os.path.join(data_dir, file)
                    self.samples.append((path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        sequence = np.load(path)
        return torch.tensor(sequence, dtype=torch.float32), label

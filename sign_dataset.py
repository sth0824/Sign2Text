# sign_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

# ✅ 1. PyTorch Dataset 클래스 만들기
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []

        # 모든 .npy 파일 경로 수집 및 라벨 추출
        for file in os.listdir(data_dir):
            if file.endswith('.npy'):
                label = file.split('_')[0]  # 예: like_like_keypoints.npy → like
                filepath = os.path.join(data_dir, file)
                self.samples.append(filepath)
                self.labels.append(label)

        # 고유 라벨 → 인덱스 숫자로 매핑
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        label = self.labels[idx]
        data = np.load(filepath)  # shape: [T, D]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(self.label2idx[label], dtype=torch.long)
        return data_tensor, label_tensor

# ✅ 2. DataLoader로 불러오기
def collate_fn(batch):
    # 시퀀스 길이가 다를 수 있으므로 pad_sequence 사용
    sequences, labels = zip(*batch)
    padded_seqs = rnn_utils.pad_sequence(sequences, batch_first=True)  # shape: [B, T_max, D]
    return padded_seqs, torch.tensor(labels)

# ✅ 3. 한 배치 확인해보기 (테스트용 실행 코드)
if __name__ == '__main__':
    data_dir = 'processed_sequences'  # npy 저장된 폴더
    dataset = SignLanguageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch_x, batch_y in dataloader:
        print("입력 시퀀스 shape:", batch_x.shape)  # [B, T_max, D]
        print("정답 라벨:", batch_y)               # [B]
        break

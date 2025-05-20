import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import re

# 1. Dataset 정의
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, label_map_path=None, min_samples=1):
        self.samples, self.labels = [], []
        self.label_mapping = {}
        if label_map_path:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
        for file in os.listdir(data_dir):
            if not file.endswith('.npy'):
                continue
            filename_no_ext = file[:-4]
            # 파일명에서 '_chunk숫자' 제거
            label_base = re.sub(r'_chunk\d+$', '', filename_no_ext)
            representative = label_base.split(',')[0]
            mapped = self.label_mapping.get(label_base, representative)
            self.samples.append(os.path.join(data_dir, file))
            self.labels.append(mapped)
        counts = Counter(self.labels)
        filtered = [(s, l) for s, l in zip(self.samples, self.labels) if counts[l] >= min_samples]
        if not filtered:
            raise ValueError(f"유효한 샘플이 없습니다. min_samples={min_samples}")
        self.samples, self.labels = zip(*filtered)
        self.mapping = {lbl: idx for idx, lbl in enumerate(sorted(set(self.labels)))}
        self.idx2label = {i: lbl for lbl, i in self.mapping.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        label = self.mapping[self.labels[idx]]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 2. Collate 함수
def collate_fn(batch):
    sequences, labels = zip(*batch)
    # 다양한 길이의 시퀀스를 같은 길이로 패딩
    padded = rnn_utils.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels, dtype=torch.long)

# 3. LSTM + Transformer 모델 정의
class SignLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
                 nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # [B, T, H] -> [T, B, H]: Transformer 입력 형식으로 변환
        trans_in = lstm_out.permute(1, 0, 2)
        trans_out = self.transformer(trans_in)
        # 시퀀스 차원 평균 풀링
        pooled = trans_out.mean(dim=0)
        return self.classifier(pooled)

# 4. 학습 실행
if __name__ == '__main__':
    data_dir = 'processed_sequences'
    label_map_path = 'label_mapping.json'
    dataset = SignLanguageDataset(data_dir, label_map_path, min_samples=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    input_size, hidden_size = 225, 128
    num_classes = len(dataset.mapping)
    model = SignLSTMTransformer(input_size, hidden_size, num_classes)
    # 모델과 데이터를 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"학습 시작: {len(dataset)} 샘플, {num_classes} 클래스")
    for epoch in range(150):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1:03d}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'sign_lstm_transformer.pth')
    print("모델 저장 완료: sign_lstm_transformer.pth")

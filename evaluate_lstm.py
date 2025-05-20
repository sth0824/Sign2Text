import os  # 파일 및 경로 관리
import numpy as np  # 시퀀스 데이터 로드용
import torch  # PyTorch 기본
from torch.utils.data import DataLoader  # 배치 처리를 위한 DataLoader
import torch.nn.utils.rnn as rnn_utils  # 가변 길이 시퀀스 패딩 지원
from train_lstm import SignLanguageDataset, SignLSTMTransformer, collate_fn  # 커스텀 데이터셋, 모델, 배치 함수

# 1) 설정
data_dir = "processed_sequences"                   # 전처리된 시퀀스 데이터 폴더
model_path = "sign_lstm_transformer.pth"           # 저장된 모델 파일 경로
input_size = 225                                    # 입력 특징 수 (예: 관절 좌표 75개 × 3차원)
hidden_size = 128                                   # LSTM 은닉 상태 차원
batch_size = 2                                      # 배치 크기

# 2) 데이터셋 및 DataLoader 구성
# SignLanguageDataset: 파일명→레이블 추출→.npy 로드→레이블 매핑→필터링 처리
dataset = SignLanguageDataset(data_dir)
# collate_fn: 배치 내 시퀀스 길이가 다를 때 pad_sequence로 패딩하여 텐서 크기 맞춤
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 3) 모델 초기화 및 파라미터 로드
# dataset.mapping: 데이터셋 생성 시 라벨 문자열→인덱스 매핑 딕셔너리
num_classes = len(dataset.mapping)
model = SignLSTMTransformer(input_size, hidden_size, num_classes)  # LSTM + Transformer 결합 모델
# map_location='cpu' 옵션으로 CPU에서 로드 (GPU 환경에서는 map_location 생략 가능)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # 평가 모드: 드롭아웃/배치정규화 비활성화

# 4) 평가 루프
correct = 0  # 정확 예측 수 누적
total = 0    # 전체 샘플 수 누적

# with torch.no_grad():
#   평가 시 불필요한 그래디언트 계산 비활성화로 메모리 절약 및 속도 개선
with torch.no_grad():
    for x_batch, y_batch in dataloader:
        # x_batch: [B, seq_len, input_size], y_batch: [B]
        outputs = model(x_batch)                # 모델 순전파: 클래스별 로그 확률(logits)
        preds = torch.argmax(outputs, dim=1)    # 가장 높은 점수(logit)를 가진 클래스 선택
        correct += (preds == y_batch).sum().item()  # 일치 개수 누적
        total += y_batch.size(0)                  # 배치 크기만큼 전체 카운트 증가

# 최종 정확도 계산 및 출력
accuracy = correct / total * 100
print(f"✅ 정확도 (Accuracy): {accuracy:.2f}%")

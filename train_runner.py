from app.train import train_model
import torch

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 사용 디바이스: {device}")

# 모델 학습 실행
model, metrics = train_model(
    data_dir="npy_data",
    label_map_path="app/label_mapping.json",
    output_model_path="trained_model.pt",
    input_size=225,  # MediaPipe 키포인트 크기
    hidden_size=128,  # LSTM 히든 크기
    batch_size=4,    # 배치 크기
    epochs=100,      # 에폭 수
    n_splits=5,      # 교차 검증 폴드 수
    device=device    # 학습 디바이스
)

print("\n🎉 학습 완료!")
print("📊 최종 평가 지표:")
for metric, value in metrics.items():
    print(f"- {metric}: {value:.4f}")

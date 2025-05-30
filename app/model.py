# 🎓 CampusSign - 통합형 수어 번역기 프로젝트 초기 템플릿

# 1. app/model.py - LSTM 모델 정의
import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))


def load_model(model_path, input_size, hidden_size, num_classes):
    model = SignLSTM(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
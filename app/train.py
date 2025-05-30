import sys
print("🔍 sys.path:", sys.path)


# app/train.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from app.dataset import SignDataset      # ✅ 요거!
from app.model import SignLSTM           # ✅ 요것도 app.model 경로에서


def plot_confusion_matrix(cm, labels, output_path):
    """혼동 행렬 시각화 함수"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, dataloader, criterion, device='cpu'):
    """모델 평가 함수"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

def train_model(data_dir, label_map_path, output_model_path, input_size=225, hidden_size=128, 
                batch_size=4, epochs=100, n_splits=5, device='cpu'):
    """
    개선된 모델 학습 및 평가 함수
    
    Args:
        data_dir: 데이터 디렉토리
        label_map_path: 라벨 매핑 파일 경로
        output_model_path: 모델 저장 경로
        input_size: 입력 크기
        hidden_size: LSTM 히든 크기
        batch_size: 배치 크기
        epochs: 에폭 수
        n_splits: 교차 검증 폴드 수 (기본값: 5)
        device: 학습 디바이스
    """
    # 라벨 매핑 로드
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    label_set = set()
    for v in label_mapping.values():
        if isinstance(v, list):
            label_set.update(v)
        else:
            label_set.add(v)
    label_list = sorted(label_set)
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for label, i in label2idx.items()}

    # 데이터셋 준비
    dataset = SignDataset(data_dir, label2idx)
    
    # 데이터셋 크기에 따라 교차 검증 폴드 수 조정
    n_samples = len(dataset)
    n_splits = min(n_splits, n_samples)  # 폴드 수를 샘플 수보다 크지 않게 조정
    
    if n_splits < 2:
        print(f"⚠️ 경고: 데이터셋 크기가 너무 작습니다 (n_samples={n_samples}).")
        print("교차 검증 대신 단일 검증 세트를 사용합니다.")
        # 단일 검증 세트 사용
        train_size = int(0.8 * n_samples)
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        train_ids, val_ids = indices[:train_size], indices[train_size:]
        
        # 데이터 로더 생성
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        # 모델 초기화
        model = SignLSTM(input_size, hidden_size, num_classes=len(label2idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 학습 루프
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # 학습
            model.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 검증
            val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
            
            # 에폭별 결과 출력
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # 최고 성능 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        # 최고 성능 모델로 평가
        model.load_state_dict(best_model_state)
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        
        # 평가 지표 계산
        metrics = classification_report(val_labels, val_preds, 
                                     target_names=[idx2label[i] for i in range(len(label2idx))],
                                     output_dict=True)
        
        # 혼동 행렬 시각화
        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(cm, [idx2label[i] for i in range(len(label2idx))],
                            'confusion_matrix_final.png')
        
        # 평균 지표 계산
        avg_metrics = {
            'precision': metrics['weighted avg']['precision'],
            'recall': metrics['weighted avg']['recall'],
            'f1-score': metrics['weighted avg']['f1-score']
        }
        
    else:
        print(f"\n📊 {n_splits}-fold 교차 검증 시작 (데이터셋 크기: {n_samples})")
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\n🔍 Fold {fold + 1}/{n_splits}")
            
            # 데이터 로더 생성
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
            
            # 모델 초기화
            model = SignLSTM(input_size, hidden_size, num_classes=len(label2idx)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 학습 루프
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(epochs):
                # 학습
                model.train()
                train_loss = 0
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 검증
                val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
                
                # 에폭별 결과 출력
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
            
            # 최고 성능 모델로 평가
            model.load_state_dict(best_model_state)
            val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
            
            # 폴드별 평가 지표 계산
            fold_metrics = classification_report(val_labels, val_preds, 
                                              target_names=[idx2label[i] for i in range(len(label2idx))],
                                              output_dict=True)
            fold_results.append(fold_metrics)
            
            # 혼동 행렬 시각화
            cm = confusion_matrix(val_labels, val_preds)
            plot_confusion_matrix(cm, [idx2label[i] for i in range(len(label2idx))],
                                f'confusion_matrix_fold_{fold+1}.png')
        
        # 전체 교차 검증 결과 평균 계산
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1-score']:
            avg_metrics[metric] = np.mean([fold['weighted avg'][metric] for fold in fold_results])
    
    print("\n📊 평가 결과 요약")
    print(f"평균 Precision: {avg_metrics['precision']:.4f}")
    print(f"평균 Recall: {avg_metrics['recall']:.4f}")
    print(f"평균 F1-score: {avg_metrics['f1-score']:.4f}")
    
    # 최종 모델 저장
    torch.save(best_model_state, output_model_path)
    print(f"\n✅ 최종 모델 저장 완료: {output_model_path}")
    
    # 전체 데이터셋에 대한 최종 평가
    final_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    final_loss, final_preds, final_labels = evaluate_model(model, final_loader, criterion, device)
    
    print("\n📈 최종 모델 평가 결과")
    print(classification_report(final_labels, final_preds,
                              target_names=[idx2label[i] for i in range(len(label2idx))]))
    
    return model, avg_metrics
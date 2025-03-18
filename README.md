# 🖐️ Sign2Text AI - 수어 인식 및 자연어 변환 AI

![License](https://img.shields.io/badge/license-MIT-green)  
![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![Deep Learning](https://img.shields.io/badge/DeepLearning-TensorFlow%20%7C%20PyTorch-orange)

## 📌 프로젝트 개요
**Sign2Text AI**는 **수어(Sign Language) 영상을 분석하여 자연어(Natural Language)로 변환**하는 인공지능 기반 프로젝트입니다.  
딥러닝과 컴퓨터 비전을 활용하여 사용자의 수어 사용 영상을 실시간으로 인식하고, **텍스트 또는 음성으로 변환**하는 기능을 제공합니다.

## 🎯 주요 기능
✅ **수어 데이터 학습**: 대규모 원천 수어 영상 데이터를 학습  
✅ **딥러닝 기반 영상 인식**: CNN, LSTM, Transformer 모델을 활용한 수어 인식  
✅ **실시간 수어 번역**: 사용자의 웹캠 또는 영상 입력을 분석하여 자연어로 변환  
✅ **다양한 언어 지원**: 한국어, 영어 등 다국어 변환 가능  

---

## 📂 데이터셋 및 학습
### 🔹 1. 원천 데이터 수집
- **공개 데이터셋 활용** (Ex: RWTH-PHOENIX-Weather 2014T, KETI 한국 수어 데이터)
- **수어 영상 데이터 라벨링** (수어 → 자연어 매핑)

### 🔹 2. 데이터 전처리
- OpenCV 및 Mediapipe를 사용한 **손 모양 및 동작 특징 추출**  
- **YOLO, MediaPipe Holistic**을 활용한 수어 주요 포인트 검출  

### 🔹 3. 모델 학습
- **CNN-LSTM 기반 모델**: 영상 특징 추출 후 LSTM을 활용한 시퀀스 학습  
- **Transformer 기반 모델**: 자연어 처리 모델과 결합하여 문장 단위 번역 수행  

---

## 🛠️ 설치 및 실행 방법
### 🔹 1. 환경 설정
```bash
git clone https://github.com/your-repo/Sign2Text-AI.git
cd Sign2Text-AI
pip install -r requirements.txt

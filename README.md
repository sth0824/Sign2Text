# 📹 CampusSign - 실시간 수어 번역기

> 대학생을 위한 상황별 수어 인식 및 자연어 문장 출력 시스템

---

## 📌 프로젝트 개요

**CampusSign**은 수어 영상을 입력받아  
- 수어 단어를 인식하고  
- 해당 의미에 맞는 자연어 문장을 출력하는  

실시간 수어 번역 웹 애플리케이션입니다.  
대학교 캠퍼스에서 자주 사용하는 수어 표현(출결, 과제, 도서관 등)에 최적화되어 있습니다.

---

## 🎯 주요 기능

- 🎥 수어 영상 업로드 (.mp4, .mkv)
- 📸 실시간 웹캠 입력 (로컬 환경 전용)
- 🧠 LSTM 기반 단어 인식
- 📄 자연어 문장 출력
- 🔍 Mediapipe 기반 키포인트 추출 (.json), 자동 전처리 (.npy)

---

## 🧩 프로젝트 구조

```
CampusSign/
├── app/
│   ├── extractor.py              ← Mediapipe 기반 키포인트 추출
│   ├── preprocess.py             ← JSON → 시퀀스 변환
│   ├── dataset.py                ← PyTorch Dataset 정의
│   ├── model.py                  ← LSTM 모델 정의 및 로딩
│   ├── train.py                  ← 모델 학습 로직
│   ├── run_pipeline.py           ← .mkv → .json → .npy 자동 변환
│   ├── label_mapping.json        ← 라벨 매핑 정보
│   └── sentence_templates.json   ← 단어별 문장 템플릿
├── train_runner.py               ← 학습 실행용 스크립트
├── streamlit_app.py              ← Streamlit 기반 웹앱 실행 파일
├── npy_data/                     ← 전처리된 .npy 데이터 저장 폴더
├── assets/sign_videos/           ← 원본 수어 영상 저장 폴더
├── requirements.txt              ← 의존성 목록
└── README.md
```


## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
````

### 2. 수어 영상 → 전처리 데이터 변환 (.mkv → .json → .npy)

```bash
python app/run_pipeline.py
```

### 3. 모델 학습 실행

```bash
python train_runner.py
```

### 4. 웹앱 실행 (Streamlit)

```bash
streamlit run streamlit_app.py
```

> ✅ 영상 업로드 모드는 웹 환경에서도 작동
> ⚠️ 실시간 웹캠 모드는 로컬(PC) 환경에서만 정상 작동

---

## 🧠 사용 기술

* **Python 3.10**
* **MediaPipe Holistic**: 키포인트 추출
* **PyTorch**: LSTM 모델 학습 및 추론
* **OpenCV**: 실시간 영상 처리
* **Streamlit**: 웹 기반 인터페이스

---

## 📝 예시 시나리오

| 상황     | 예시 단어   | 출력 문장 예시          |
| ------ | ------- | ----------------- |
| 출결     | 지각, 출석  | 지각했습니다. 출석 가능할까요? |
| 수업 요청  | 질문, 다시  | 이 부분 다시 설명해 주세요.  |
| 과제 안내  | 과제, 제출  | 과제 제출일이 언제인가요?    |
| 캠퍼스 생활 | 식당, 도서관 | 도서관은 어디에 있어요?     |

---

## 🔧 향후 개발 계획

* 🔁 실시간 다단어 → 문장 인식 고도화 (예: "지각 + 출석")
* ☁️ Streamlit Cloud 배포 최적화
* 🧪 사용자 정의 수어 추가 및 학습 기능
* 🔊 음성 출력 연동 (TTS)


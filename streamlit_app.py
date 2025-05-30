import streamlit as st
import numpy as np
import tempfile
import os
import cv2
import json
import torch
import shutil

from app.extractor import extract_keypoints_from_video
from app.preprocess import convert_json_to_sequence
from app.model import load_model

# 설정
MODEL_PATH = "trained_model.pt"
LABEL_MAP_PATH = "app/label_mapping.json"
SENTENCE_TEMPLATE_PATH = "app/sentence_templates.json"
INPUT_SIZE = 225
HIDDEN_SIZE = 128

# 모델 로딩
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    label_mapping = json.load(f)

label_set = set()
for v in label_mapping.values():
    label_set.update(v) if isinstance(v, list) else label_set.add(v)
label_list = sorted(label_set)
label2idx = {label: i for i, label in enumerate(label_list)}
idx2label = {i: label for label, i in label2idx.items()}
num_classes = len(label2idx)

model = load_model(MODEL_PATH, INPUT_SIZE, HIDDEN_SIZE, num_classes)

# Streamlit UI

# --- 커스텀 CSS 스타일 추가 ---
st.markdown("""
<style>
/* 전체 페이지 기본 글꼴 및 배경 색상 */
body {
    font-family: 'Nanum Gothic', sans-serif; /* 원하는 글꼴로 변경하세요 */
    background-color: #1e1e1e; /* 다크 모드 배경 색상 */
    color: #ffffff; /* 기본 글자 색상 */
}

/* Streamlit 제목 스타일 */
h1 {
    color: #4CAF50; /* 제목 색상 */
}

/* 섹션 제목 (st.write 등) */
.stMarkdown {
    color: #cccccc; /* 섹션 설명 색상 */
}

/* 컨테이너 스타일 */
.container {
    background-color: #2d2d2d; /* 컨테이너 배경 색상 */
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

/* 파일 업로더 스타일 */
.stFileUploader {
    background-color: #3a3a3a;
    padding: 15px;
    border-radius: 8px;
}

/* 라디오 버튼 스타일 */
.stRadio > label > div {
    color: #ffffff; /* 라디오 버튼 텍스트 색상 */
}

/* 성공/정보/경고 메시지 스타일 */
.stAlert {
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
}
.stAlert.stSuccess { background-color: #28a74533; color: #28a745; } /* 연한 초록 배경 */
.stAlert.stInfo { background-color: #17a2b833; color: #17a2b8; }    /* 연한 파랑 배경 */
.stAlert.stWarning { background-color: #ffc10733; color: #ffc107; } /* 연한 노랑 배경 */
.stAlert.stError { background-color: #dc354533; color: #dc3545; }   /* 연한 빨강 배경 */


/* 버튼 스타일 */
.stButton > button {
    background-color: #4CAF50; /* 버튼 배경 색상 */
    color: white; /* 버튼 글자 색상 */
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #45a049; /* 호버 시 배경 색상 */
}

/* 로딩 스피너 스타일 */
.stSpinner > div > div {
    border-top-color: #4CAF50 !important; /* 스피너 색상 */
}

</style>
""", unsafe_allow_html=True)
# --- 커스텀 CSS 스타일 추가 끝 ---


st.title("📹 CampusSign 실시간 수어 번역기")
st.write("🟢 웹캠/영상 입력 → 수어 단어 인식 → 자연어 문장 출력")

# 입력 방식 선택 섹션 (컨테이너 사용)
with st.container():
    st.subheader("입력 방식 선택") # 작은 제목 추가
    mode = st.radio("", ["🎥 영상 업로드", "📸 실시간 웹캠 (데모용)"]) # 라디오 버튼 레이블 제거

# 템플릿 로딩
with open(SENTENCE_TEMPLATE_PATH, encoding="utf-8") as f:
    templates = json.load(f)

predicted_words = []

def predict_from_video(video_path):
    json_path = video_path.replace(".mp4", ".json").replace(".mkv", ".json")
    try:
        extract_keypoints_from_video(video_path, json_path)
    except Exception as e:
        st.error(f"❌ 키포인트 추출 중 오류: {e}")
        return None

    if not os.path.exists(json_path):
        st.error("❌ .json 파일이 생성되지 않았습니다.")
        return None

    try:
        with open(json_path, encoding='utf-8') as f:
            kp_data = json.load(f)
        sequence_array = convert_json_to_sequence(kp_data)
        if sequence_array is None:
            st.error("❌ 키포인트 시퀀스가 비어있거나 변환 실패.")
            return None
    except Exception as e:
        st.error(f"❌ JSON 처리 중 오류 발생: {e}")
        return None

    try:
        x = torch.tensor(sequence_array[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x)
            # Softmax를 적용하여 확률 분포를 얻고, 가장 높은 확률의 인덱스를 선택합니다.
            # Streamlit 앱의 원래 코드에는 softmax가 명시적으로 없었지만, 예측 로직에 따라 추가될 수 있습니다.
            # 여기서는 argmax만 사용하여 가장 가능성 높은 클래스 인덱스를 가져옵니다.
            pred_label_idx = torch.argmax(y_pred).item()
            if pred_label_idx < len(idx2label): # 인덱스가 유효한 범위 내에 있는지 확인
                 pred_label = idx2label[pred_label_idx]
                 return pred_label
            else:
                 st.warning(f"❗ 예측 인덱스 범위 오류: {pred_label_idx}")
                 return None

    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")
        return None


# 🎥 영상 업로드 모드 (컨테이너 사용 및 로딩 스피너 추가)
if mode == "🎥 영상 업로드":
    with st.container():
        st.subheader("수어 영상 업로드") # 작은 제목 추가
        uploaded_file = st.file_uploader("(.mp4, .mkv)", type=["mp4", "mkv"])
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                shutil.copyfileobj(uploaded_file, temp_video)
                temp_path = temp_video.name

            # 로딩 스피너 추가
            with st.spinner("⏳ 영상 처리 및 예측 중..."):
                pred = predict_from_video(temp_path)

            # 예측 결과 섹션 (컨테이너 사용)
            with st.container():
                st.subheader("⭐ 예측 결과") # 작은 제목 추가
                if pred:
                    st.success(f"🧠 예측된 단어: **{pred}**")
                    # 변경 시작: 중첩된 템플릿 구조에서 단어 찾기
                    sentence = None
                    for category, words in templates.items():
                        if pred in words:
                            sentence = words[pred]
                            break # 찾았으면 반복 중단
                    # 변경 끝

                    if sentence:
                        st.info(f"📝 출력 문장: {sentence}")
                    else:
                        st.warning(f"ℹ️ \'{pred}\'에 대한 문장 템플릿이 없습니다.")
                else:
                    st.warning("❗ 예측 결과가 없습니다.")
            # 예측 결과 섹션 (컨테이너 사용 끝)

# 📸 실시간 웹캠 모드 (컨테이너 사용 및 로딩 스피너 추가)
elif mode == "📸 실시간 웹캠 (데모용)":
     with st.container():
        st.subheader("실시간 웹캠") # 작은 제목 추가
        st.warning("⚠️ 이 모드는 데스크탑에서만 작동하며, Streamlit Cloud에서는 지원되지 않습니다.")
        run_webcam = st.button("실시간 예측 시작")
        if run_webcam:
            st.info("📸 웹캠을 열고 한 단어 수어를 보여주세요. 예: \'출석\', \'과제\' 등")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ 웹캠을 열 수 없습니다. 다른 앱이 사용 중일 수 있습니다.")
            else:
                frame_list = []
                # 로딩 스피너 추가
                with st.spinner("⏳ 웹캠 영상 수집 중 (약 2초)..."):
                    for _ in range(60):  # 약 2초간 프레임 수집
                        ret, frame = cap.read()
                        if not ret:
                            st.error("웹캠에서 프레임을 가져올 수 없습니다.")
                            break
                        frame = cv2.flip(frame, 1)
                        frame_list.append(frame)
                cap.release()

                if frame_list:
                    temp_path = os.path.join(tempfile.gettempdir(), "webcam_clip.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # VideoWriter 설정 시 프레임 크기 확인
                    if frame_list:
                        height, width, _ = frame_list[0].shape
                        out = cv2.VideoWriter(temp_path, fourcc, 15.0, (width, height))
                        for frame in frame_list:
                            out.write(frame)
                        out.release()
                    else:
                         st.error("❌ 수집된 웹캠 프레임이 없습니다.")
                         temp_path = None # 예측 단계로 넘어가지 않도록 경로 초기화

                    if temp_path and os.path.exists(temp_path):
                         # 로딩 스피너 추가
                        with st.spinner("⏳ 영상 처리 및 예측 중..."):
                             pred = predict_from_video(temp_path)
                    else:
                         pred = None

                    # 예측 결과 섹션 (컨테이너 사용)
                    with st.container():
                        st.subheader("⭐ 예측 결과") # 작은 제목 추가
                        if pred:
                            predicted_words.append(pred)
                            st.success(f"🧠 예측된 단어: **{pred}**")
                            st.write(f"🧩 현재 인식된 단어들: `{predicted_words}`") # 현재까지 인식된 단어 목록 표시

                            # 중첩된 템플릿 구조에서 단어 찾기 (이전 단계에서 수정됨)
                            sentence = None
                            for category, words in templates.items():
                                if pred in words:
                                    sentence = words[pred]
                                    break

                            if sentence:
                                st.info(f"📝 출력 문장 예시: {sentence}")
                            else:
                                st.warning(f"ℹ️ \'{pred}\'에 대한 문장 템플릿이 없습니다.")
                        else:
                            st.warning("❗ 예측 결과가 없습니다.")
                    # 예측 결과 섹션 (컨테이너 사용 끝)
# 📸 실시간 웹캠 모드 (컨테이너 사용 및 로딩 스피너 추가 끝)

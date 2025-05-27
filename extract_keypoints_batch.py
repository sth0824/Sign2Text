# extract_keypoints_batch.py
import os
import cv2
import json
import mediapipe as mp

# 입력/출력 디렉토리 설정
input_dir = 'source_videos_50'
output_dir = 'keypoint_jsons'
os.makedirs(output_dir, exist_ok=True)

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# 키포인트 추출 함수
def extract_landmarks(landmarks):
    if landmarks:
        return [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in landmarks.landmark]
    else:
        return []

# 지원하는 확장자 목록 (.mp4, .mkv)
valid_exts = ['.mkv', '.mp4']

# 모든 동영상 처리
for filename in os.listdir(input_dir):
    if not any(filename.endswith(ext) for ext in valid_exts):
        continue

    video_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_json = os.path.join(output_dir, f'{base_name}.json')

    cap = cv2.VideoCapture(video_path)
    frame_data = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        keypoints = {
            "pose": extract_landmarks(results.pose_landmarks),
            "face": extract_landmarks(results.face_landmarks),
            "left_hand": extract_landmarks(results.left_hand_landmarks),
            "right_hand": extract_landmarks(results.right_hand_landmarks),
        }

        frame_data[f'frame_{frame_idx:05d}'] = keypoints
        frame_idx += 1

    cap.release()

    # JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(frame_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 추출 완료: {output_json}")

# 종료
holistic.close()

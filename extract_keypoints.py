import cv2
import mediapipe as mp
import json
import os
import glob

# 원본 영상 경로
video_path = 'like.mkv'

# ▶️ base_name: like
base_name = os.path.splitext(os.path.basename(video_path))[0]

# 시각화 결과 영상 경로
output_image_dir = 'output_frames'
output_json_path = f'{base_name}_keypoints.json'
output_video_path = f'{base_name}_result.mp4'
comparison_video_path = f'{base_name}_comparison.mp4'

# 출력 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
holistic = mp_holistic.Holistic(static_image_mode=False)

# 스타일 지정
red_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)    # 손: 빨강
blue_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)   # 얼굴: 파랑

# 영상 열기
cap = cv2.VideoCapture(video_path)
frame_data = {}
frame_idx = 0

# 키포인트 좌표 추출 함수
def extract_landmarks(landmarks):
    if landmarks:
        return [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]
                for lm in landmarks.landmark]
    else:
        return []

# 영상 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # 키포인트 저장
    keypoints = {
        "pose": extract_landmarks(results.pose_landmarks),
        "face": extract_landmarks(results.face_landmarks),
        "left_hand": extract_landmarks(results.left_hand_landmarks),
        "right_hand": extract_landmarks(results.right_hand_landmarks),
    }
    frame_data[f'frame_{frame_idx:05d}'] = keypoints

    # 시각화
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=blue_spec,
            connection_drawing_spec=blue_spec
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=red_spec,
            connection_drawing_spec=red_spec
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=red_spec,
            connection_drawing_spec=red_spec
        )

    # 이미지 저장
    frame_filename = os.path.join(output_image_dir, f'frame_{frame_idx:05d}.png')
    cv2.imwrite(frame_filename, image_bgr)

    frame_idx += 1

# 자원 해제
cap.release()
holistic.close()

# JSON 저장
with open(output_json_path, 'w') as f:
    json.dump(frame_data, f, indent=2)

print(f"✅ 키포인트 JSON 저장 완료: {output_json_path}")

# ------------------ 프레임 → 시각화 영상 ------------------
image_files = sorted(glob.glob(os.path.join(output_image_dir, '*.png')))
if not image_files:
    raise ValueError("❌ output_frames 폴더에 이미지가 없습니다!")

first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for file in image_files:
    img = cv2.imread(file)
    video_writer.write(img)

video_writer.release()
print(f"🎬 시각화 영상 저장 완료: {output_video_path}")

# ------------------ 원본 + 시각화 비교 영상 ------------------
cap_orig = cv2.VideoCapture(video_path)
cap_vis = cv2.VideoCapture(output_video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(comparison_video_path, fourcc, fps, (width * 2, height))

while True:
    ret1, frame_orig = cap_orig.read()
    ret2, frame_vis = cap_vis.read()

    if not ret1 or not ret2:
        break

    combined = cv2.hconcat([frame_orig, frame_vis])
    out.write(combined)

cap_orig.release()
cap_vis.release()
out.release()

print(f"📹 비교 영상 저장 완료: {comparison_video_path}")

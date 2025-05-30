# app/extractor.py
import cv2
import mediapipe as mp
import json
import os

def extract_keypoints_from_video(video_path, output_json_path):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)

    def extract_landmarks(landmarks):
        if landmarks:
            return [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in landmarks.landmark]
        else:
            return []

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
    holistic.close()

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(frame_data, f, indent=2, ensure_ascii=False)

    return output_json_path
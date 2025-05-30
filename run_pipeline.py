# run_pipeline.py

import os
from app.extractor import extract_keypoints_from_video
from app.preprocess import process_folder

# 설정
video_dir = "assets/sign_videos"
json_dir = "extracted_json"
npy_dir = "npy_data"
label_mapping_path = "app/label_mapping.json"

# Step 1: .mkv → .json 변환
os.makedirs(json_dir, exist_ok=True)
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mkv")]

print("📦 [1단계] .mkv → .json 변환 시작")
for filename in video_files:
    input_path = os.path.join(video_dir, filename)
    output_path = os.path.join(json_dir, filename.replace(".mkv", ".json"))
    print(f"🔍 추출 중: {filename}")
    extract_keypoints_from_video(input_path, output_path)
    print(f"✅ 저장 완료: {output_path}")

# Step 2: .json → .npy 변환
print("\n📦 [2단계] .json → .npy 변환 시작")
process_folder(
    json_dir=json_dir,
    output_dir=npy_dir,
    label_mapping_path=label_mapping_path
)
print(f"✅ 모든 .npy 저장 완료: {npy_dir}")

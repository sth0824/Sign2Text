import os
import json
import numpy as np
from tqdm import tqdm

# 🔧 설정
json_dir = 'keypoint_jsons'         # 입력 JSON 폴더
output_dir = 'processed_sequences'  # 출력 .npy 폴더
label_map_file = 'label_mapping.json'
min_frames = 30                     # 너무 짧은 경우 복제 기준
target_frames = 60                 # 타겟 프레임 길이
stride = 30                        # 슬라이딩 윈도우 보간 간격

os.makedirs(output_dir, exist_ok=True)

# 사용할 keypoint 부위
USE_PARTS = ['left_hand', 'right_hand', 'pose']
PART_COUNTS = {'left_hand': 21, 'right_hand': 21, 'pose': 33, 'face': 468}
EXPECTED_DIM = sum(PART_COUNTS[p] for p in USE_PARTS) * 3  # x, y, z

# 라벨 매핑 불러오기
with open(label_map_file, 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

def flatten_keypoints_with_padding(kp_dict, parts=USE_PARTS):
    flat = []
    for part in parts:
        keypoints = kp_dict.get(part, [])
        if keypoints:
            for point in keypoints:
                flat.extend(point[:3])
            missing = PART_COUNTS[part] - len(keypoints)
            flat.extend([0.0] * missing * 3)
        else:
            flat.extend([0.0] * PART_COUNTS[part] * 3)
    return flat

# 변환 루프
for json_file in tqdm(os.listdir(json_dir)):
    if not json_file.endswith('.json'):
        continue

    raw_label = json_file.replace('.json', '').split('_')[-1]
    representative_label = label_mapping.get(raw_label, raw_label)

    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    sequence = []
    for frame_key in sorted(data.keys()):
        flat_kp = flatten_keypoints_with_padding(data[frame_key])
        sequence.append(flat_kp)

    if len(sequence) < min_frames:
        print(f"⚠️ 너무 짧음 → 복제: {json_file} ({len(sequence)} frames)")
        reps = int(np.ceil(target_frames / len(sequence)))
        sequence = (sequence * reps)[:target_frames]
        sequence_array = np.array(sequence, dtype=np.float32)
        save_name = f'{representative_label}_{raw_label}_padded.npy'
        np.save(os.path.join(output_dir, save_name), sequence_array)
        print(f"✅ 저장 완료 (복제): {save_name} ({sequence_array.shape})")

    elif len(sequence) <= target_frames:
        sequence_array = np.array(sequence, dtype=np.float32)
        save_name = f'{representative_label}_{raw_label}.npy'
        np.save(os.path.join(output_dir, save_name), sequence_array)
        print(f"✅ 저장 완료: {save_name} ({sequence_array.shape})")

    else:
        total_frames = len(sequence)
        for start in range(0, total_frames - target_frames + 1, stride):
            chunk = sequence[start:start + target_frames]
            sequence_array = np.array(chunk, dtype=np.float32)
            save_name = f'{representative_label}_{raw_label}_chunk{start}.npy'
            np.save(os.path.join(output_dir, save_name), sequence_array)
            print(f"✅ 저장 완료 (윈도우): {save_name} ({sequence_array.shape})")

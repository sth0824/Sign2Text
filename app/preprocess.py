import os
import json
import numpy as np

# 사용할 키포인트 종류
USE_PARTS = ['left_hand', 'right_hand', 'pose']
PART_COUNTS = {'left_hand': 21, 'right_hand': 21, 'pose': 33, 'face': 468}

def flatten_keypoints_with_padding(kp_dict, parts=USE_PARTS):
    """
    각 프레임에서 필요한 파트의 키포인트를 [x, y, z]로 평탄화하고,
    부족한 포인트는 0.0으로 패딩
    """
    flat = []
    for part in parts:
        keypoints = kp_dict.get(part, [])
        if keypoints:
            for point in keypoints:
                flat.extend(point[:3])  # x, y, z만 사용
            missing = PART_COUNTS[part] - len(keypoints)
            flat.extend([0.0] * missing * 3)
        else:
            flat.extend([0.0] * PART_COUNTS[part] * 3)
    return flat

def convert_json_to_sequence(json_data, min_frames=30, target_frames=60):
    """
    JSON 데이터(프레임별 키포인트)를 시퀀스 배열로 변환.
    - 프레임 수가 부족하면 None 반환
    - 프레임 수가 target_frames보다 적으면 반복 복제
    - 초과하면 잘라냄
    """
    sequence = []
    for frame_key in sorted(json_data.keys()):
        flat_kp = flatten_keypoints_with_padding(json_data[frame_key])
        sequence.append(flat_kp)

    if len(sequence) < min_frames:
        return None

    if len(sequence) >= target_frames:
        sequence = sequence[:target_frames]
    else:
        reps = int(np.ceil(target_frames / len(sequence)))
        sequence = (sequence * reps)[:target_frames]

    return np.array([sequence], dtype=np.float32)

def process_folder(json_dir, output_dir, label_mapping_path, min_frames=30, target_frames=60):
    """
    폴더 단위로 .json → .npy 시퀀스 변환
    - label_mapping.json 참고하여 라벨 매핑
    - 한 개의 npy 시퀀스가 만들어짐
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        # 파일명에서 라벨 추출
        raw_label = filename.replace('.json', '').split('_')[-1]
        mapped_label = label_mapping.get(raw_label, raw_label)

        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sequence_array = convert_json_to_sequence(data, min_frames, target_frames)
        if sequence_array is not None:
            # ✅ 라벨이 리스트일 경우 첫 번째 요소만 사용
            label_name = mapped_label[0] if isinstance(mapped_label, list) else mapped_label
            save_path = os.path.join(output_dir, f"{label_name}.npy")
            np.save(save_path, sequence_array[0])
            print(f"✅ 저장 완료: {save_path}")
            
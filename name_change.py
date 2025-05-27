import os
import json

data_dir = "processed_sequences"
output_mapping_file = "label_mapping.json"

def extract_representative_label(label_string):
    return label_string.split(',')[0].strip()

label_mapping = {}
renamed_files = []

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        original_path = os.path.join(data_dir, filename)
        label_part = filename.replace(".npy", "").split('_')[-1]
        representative_label = extract_representative_label(label_part)

        new_filename = filename.replace(label_part, representative_label)
        new_path = os.path.join(data_dir, new_filename)

        # 변경 이름이 다르고, 같은 이름의 파일이 아직 존재하지 않을 때만 수행
        if new_filename != filename and not os.path.exists(new_path):
            os.rename(original_path, new_path)
            label_mapping[label_part] = representative_label
            renamed_files.append((filename, new_filename))
        elif new_filename != filename and os.path.exists(new_path):
            print(f"⚠️ 이미 존재하여 건너뜀: {new_filename}")

# 매핑 저장
with open(output_mapping_file, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

print("\n✅ 파일 이름 변경 완료")
for old, new in renamed_files:
    print(f"{old} → {new}")

import os
import pandas as pd
import shutil

# Paths adjusted to match your exact directory image layout
DATA_DIR = os.path.join("data", "Indian_Dataset")
CSV_PATH = os.path.join(DATA_DIR, "traffic_sign.csv")

print("[INFO] Starting automated Indian Road Dataset sorting script...")

if not os.path.exists(CSV_PATH):
    print(f"[ERROR] Could not locate traffic_sign.csv at expected path: {CSV_PATH}")
    exit()

# Load mapping matrix
df = pd.read_csv(CSV_PATH)

sorted_count = 0
missing_count = 0

for index, row in df.iterrows():
    # Matches your column headers: 'Path' (filename) and 'ClassId'
    img_name = str(row['Path']).strip()
    class_id = str(row['ClassId']).strip()
    
    # Path coordinates
    src_path = os.path.join(DATA_DIR, img_name)
    dest_dir = os.path.join(DATA_DIR, class_id)
    
    # Create target directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, img_name)

    # Move operation execution block
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        sorted_count += 1
    else:
        # Prevents double-move failure flags if script is executed twice
        if not os.path.exists(dest_path):
            missing_count += 1

print("\n" + "="*50)
print("[SUCCESS] Data Sorting Operation Complete!")
print(f" -> Files successfully categorized into class folders: {sorted_count}")
if missing_count > 0:
    print(f" -> Referenced images missing from root folder: {missing_count}")
print("="*50)
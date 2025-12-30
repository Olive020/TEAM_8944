import os
import cv2
import numpy as np
import glob
from pathlib import Path 

# --- 設定參數 ---
# 影像的根目錄（包含所有 patientXXXX 資料夾）
IMAGE_ROOT = r"C:\Users\Lab902\Desktop\week8\pc\data\training_image"
# 標註檔的根目錄（包含所有 patientXXXX 資料夾）
LABEL_ROOT = r"C:\Users\Lab902\Desktop\week8\pc\data\training_label"
# 輸出影像的目錄 (已修改為 combine_output)
OUTPUT_DIR = "./combine_output" 

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 顏色定義 (BGR 格式)
COLOR_RED = (0, 0, 255) # 邊界框顏色：紅色

# --- 步驟一：讀取 YOLO 標註並繪製 ---

def draw_yolo_boxes(image, label_path):
    """
    從 YOLO 標註檔中讀取座標 (歸一化 x, y, w, h)，並繪製到影像上。
    """
    
    # 獲取影像的寬度和高度
    H, W, _ = image.shape
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            # 標籤檔是空的 (純背景的負面樣本)
            cv2.putText(image, "TRUE NEGATIVE (TN)", (10, H - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return image
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # 1. 提取 YOLO 歸一化座標
            # class, x_center, y_center, width, height
            try:
                _, x_c, y_c, w, h = map(float, parts)
            except ValueError:
                continue # 忽略無效的座標行
            
            # 2. 轉換為絕對像素座標 (xyxy 格式)
            x1 = int((x_c - w / 2) * W)
            y1 = int((y_c - h / 2) * H)
            x2 = int((x_c + w / 2) * W)
            y2 = int((y_c + h / 2) * H)
            
            # 3. 繪製邊界框
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_RED, 2)
            
            # (可選) 繪製類別標籤
            class_id = int(parts[0])
            cv2.putText(image, f"Class {class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

    except FileNotFoundError:
        # 找不到標籤檔
        cv2.putText(image, "LABEL MISSING", (10, H - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    except Exception as e:
        # 處理其他錯誤
        cv2.putText(image, f"Error: {e}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    return image

# --- 步驟二：遍歷所有子資料夾並輸出 ---

print(f"正在處理影像... 輸出將保存到 {OUTPUT_DIR}")

# 使用 os.walk 遞迴搜索所有 .png 檔案
all_image_paths = []
for dirpath, dirnames, filenames in os.walk(IMAGE_ROOT):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_image_paths.append(os.path.join(dirpath, filename))

if not all_image_paths:
    print("警告：在指定的根目錄中找不到任何影像檔案。請檢查 IMAGE_ROOT 變數。")
    exit()

print(f"總共找到 {len(all_image_paths)} 張影像進行視覺化。")

for img_path in all_image_paths:
    # 獲取檔案名和路徑資訊
    base_name = os.path.basename(img_path)
    file_stem = os.path.splitext(base_name)[0]
    
    # 根據影像路徑推斷標籤路徑
    relative_path = os.path.relpath(img_path, IMAGE_ROOT) 
    
    # 組合標籤路徑: LABEL_ROOT / patientXXXX / file.txt
    label_path = os.path.join(LABEL_ROOT, relative_path).replace(Path(img_path).suffix, '.txt')
    
    # 設置輸出檔名 (格式: patientXXXX_file_labeled.png)
    patient_folder = os.path.basename(os.path.dirname(img_path))
    output_filename = f"{patient_folder}_{file_stem}_labeled.png" 
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 讀取影像
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"警告：無法讀取影像 {img_path}")
        continue
        
    # 繪製標註框
    visualized_image = draw_yolo_boxes(image, label_path)
    
    # 儲存結果
    cv2.imwrite(output_path, visualized_image)

print(f"\n✅ 所有影像處理完畢。請查看 {OUTPUT_DIR} 資料夾。")
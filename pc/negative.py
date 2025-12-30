import os
import cv2
import numpy as np
import glob
from random import randint
import shutil

# --- 設定參數 (已激進調整) ---
# 原始訓練影像的根目錄
IMAGE_DIR = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\images"
# 原始標註檔的根目錄
LABEL_DIR = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\labels"

# 這是新的輸出目錄，位於 C:\Users\Lab902\Desktop\week8\pc\datasets\train 下
OUTPUT_NEG_DIR_IMG = "./datasets/train/neg_images" 
OUTPUT_NEG_DIR_LBL = "./datasets/train/neg_labels" 

# 裁切參數
CROPS_PER_IMAGE = 30   # 提高每張圖片的裁切目標數量 (原 20 -> 30)
CROP_SIZE = 160        # 激進降低裁切尺寸，確保找到純背景 (原 320 -> 160)

# --- 步驟零：清理舊的輸出目錄 ---
if os.path.exists(OUTPUT_NEG_DIR_IMG):
    shutil.rmtree(OUTPUT_NEG_DIR_IMG)
if os.path.exists(OUTPUT_NEG_DIR_LBL):
    shutil.rmtree(OUTPUT_NEG_DIR_LBL)
    
os.makedirs(OUTPUT_NEG_DIR_IMG, exist_ok=True)
os.makedirs(OUTPUT_NEG_DIR_LBL, exist_ok=True)

# --- 步驟一：定義輔助函數 ---
def get_absolute_boxes(label_path, W, H):
    """從 YOLO 標籤檔中讀取並轉換為絕對像素座標 (xyxy 格式)"""
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                try:
                    _, x_c, y_c, w, h = map(float, parts)
                except ValueError:
                    continue 

                # 轉換為絕對像素座標
                x1 = int((x_c - w / 2) * W)
                y1 = int((y_c - h / 2) * H)
                x2 = int((x_c + w / 2) * W)
                y2 = int((y_c + h / 2) * H)
                bboxes.append((x1, y1, x2, y2))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"解析標籤失敗: {e}")
        
    return bboxes

def is_overlap(crop_x1, crop_y1, crop_x2, crop_y2, bx1, by1, bx2, by2):
    """檢查兩個矩形是否重疊"""
    return not (crop_x2 <= bx1 or crop_x1 >= bx2 or crop_y2 <= by1 or crop_y1 >= by2)

# --- 步驟二：生成負面樣本 ---

def create_negative_crops():
    print(f"開始從 {IMAGE_DIR} 創建負面樣本 (CROP_SIZE={CROP_SIZE}, 目標 {3262} 筆)...")
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    total_crops = 0
    
    # 設置總目標和單圖最大嘗試次數
    TARGET_TOTAL_CROPS = 3262 
    MAX_ATTEMPTS_PER_IMAGE = CROPS_PER_IMAGE * 10 

    for img_path in image_files:
        if total_crops >= TARGET_TOTAL_CROPS:
            break
            
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        H, W, _ = image.shape
        if W < CROP_SIZE or H < CROP_SIZE: continue # 影像太小，跳過
            
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, file_stem + ".txt")
        
        bboxes = get_absolute_boxes(label_path, W, H)

        # 隨機裁切並檢查是否與標註框重疊
        crops_created_for_image = 0
        attempts = 0
        
        while (crops_created_for_image < CROPS_PER_IMAGE and 
               attempts < MAX_ATTEMPTS_PER_IMAGE and
               total_crops < TARGET_TOTAL_CROPS):
            
            attempts += 1
            
            # 隨機選擇左上角座標
            x_start = randint(0, W - CROP_SIZE)
            y_start = randint(0, H - CROP_SIZE)
            
            x_end = x_start + CROP_SIZE
            y_end = y_start + CROP_SIZE
            
            is_overlap_flag = False
            for bx1, by1, bx2, by2 in bboxes:
                # 檢查裁切區域是否與任何標註框重疊
                if is_overlap(x_start, y_start, x_end, y_end, bx1, by1, bx2, by2):
                    is_overlap_flag = True
                    break
            
            if not is_overlap_flag:
                # 保存為負面樣本
                crop = image[y_start:y_end, x_start:x_end]
                new_filename = f"{file_stem}_neg_{crops_created_for_image}.png"
                
                cv2.imwrite(os.path.join(OUTPUT_NEG_DIR_IMG, new_filename), crop)
                
                # 創建空的標籤檔 (關鍵！)
                with open(os.path.join(OUTPUT_NEG_DIR_LBL, new_filename.replace('.png', '.txt')), 'w') as f:
                    pass
                
                total_crops += 1
                crops_created_for_image += 1
        
    print(f"\n✅ 成功創建 {total_crops} 個純背景負面樣本 (目標 {TARGET_TOTAL_CROPS} 筆)。")
    print("\n★★★ 請執行以下步驟完成數據整合 ★★★")
    print(f"1. 複製 {OUTPUT_NEG_DIR_IMG} 內容到 {IMAGE_DIR}")
    print(f"2. 複製 {OUTPUT_NEG_DIR_LBL} 內容到 {LABEL_DIR}")

if __name__ == "__main__":
    create_negative_crops()
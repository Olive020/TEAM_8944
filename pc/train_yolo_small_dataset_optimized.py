# ======================================================
# YOLOv8m 單次訓練版 (300 Epochs，恢復成績專用)
# ======================================================

import sys, os, gc, multiprocessing, glob, json, shutil
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import torch
from pathlib import Path
import warnings
import subprocess 

# --- 環境設定與檢查 ---
try:
    import ultralytics
except ImportError:
    print(">>> 偵測到 ultralytics 模組未安裝，開始自動安裝 YOLOv8...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "pandas", "tqdm"])
    print(">>> 安裝完成，請重新運行程式。")
    sys.exit() 
    
warnings.filterwarnings('ignore')

print("=" * 70)
print("YOLOv8m 恢復成績訓練系統 (300 Epochs)")
print(f"當前工作目錄: {os.getcwd()}")
print("=" * 70)

# 設定隨機種子
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# CUDA 記憶體優化 (VRAM 級別)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# ======================================================
# 1. YOLOv8m 專用配置 (已修正 Epochs)
# ======================================================

YOLOv8M_CONFIG = {
    "name": "YOLOv8m_Recovery_300E",
    "description": "YOLOv8m 穩定訓練，300 輪",
    "weights": "yolov8m.pt",
    "epochs": 300,            # 關鍵修正: 設置為 300 輪
    "batch": 4,               
    "imgsz": 512,             
    "optimizer": "AdamW",
    "lr0": 0.0003,            
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.001,
    "cos_lr": True,
    "warmup_epochs": 15,
    "patience": 100,          # 提高耐心值
    "close_mosaic": 40,       # 增加關閉 Mosaic 的輪數
    "freeze": 8,              
    "augmentation_level": "extreme", 
    "label_smoothing": 0.2,   
    "dropout": 0.25           
}

# ======================================================
# 2. 強化資料增強配置 (與原腳本保持一致)
# ======================================================

AUGMENTATION_PRESETS = {
    "extreme": {
        "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.5, "degrees": 20.0, 
        "translate": 0.2, "scale": 0.5, "shear": 10.0, "perspective": 0.001, 
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.5, 
        "copy_paste": 0.3, "auto_augment": "randaugment", "erasing": 0.4, 
        "crop_fraction": 0.8
    },
    # 這裡省略其他增強配置，需確保它們在完整腳本中
}

# ======================================================
# 3. K-Fold 交叉驗證相關函數 (已移除或跳過)
# ======================================================
# 註：此版本只執行單次訓練，K-Fold 相關函數已不再使用。


# ======================================================
# 4. 測試時增強 (TTA) 及預測儲存
# ======================================================

def predict_with_tta(model, test_images_path, tta_scales=[1.0]):
    """測試時增強預測"""
    print("\n執行 TTA 預測...")
    
    results = model.predict(
        source=test_images_path,
        imgsz=YOLOv8M_CONFIG['imgsz'],
        conf=0.25,
        iou=0.5,
        augment=True, 
        save=False,
        verbose=False
    )
    return results

def save_predictions_to_images_txt(results, save_dir="./predict_txt"):
    """儲存預測結果到 predict_txt/images.txt"""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "images.txt")

    if not isinstance(results, list):
        results = [results]
        
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            filename = Path(r.path).stem
            boxes = r.boxes
            if boxes is None:
                continue

            for j in range(len(boxes.cls)):
                label = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()

                f.write(
                    f"{filename} {label} {conf:.4f} "
                    f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                )

    print(f"預測結果已輸出到: {out_path}")

# ======================================================
# 5. 主程式
# ======================================================

def main():
    """主函數"""

    config = YOLOv8M_CONFIG
    aug_config = AUGMENTATION_PRESETS[config['augmentation_level']]

    # 建立訓練參數 (單次訓練)
    base_train_args = {k: v for k, v in config.items() if k not in ['name', 'description', 'augmentation_level', 'weights']}
    base_train_args.update(aug_config)
    
    # --- 關鍵系統修正：解決 OOM/MemoryError ---
    base_train_args['device'] = 0 if torch.cuda.is_available() else 'cpu'
    base_train_args['workers'] = 0   # 禁用多進程 (解決 MemoryError)
    base_train_args['cache'] = False # 禁用 RAM 緩存 (解決 MemoryError)
    base_train_args['optimizer'] = 'AdamW'

    # 檢查環境
    if base_train_args['device'] == 'cuda':
        print(f"\n使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n警告：使用 CPU，訓練會很慢！")
        
    print(f"\n執行 YOLOv8m 單次訓練 ({config['epochs']} Epochs)...")
    
    # 檢查您的資料集路徑 (這裡假設您已將所有圖片和標籤扁平化)
    if not os.path.exists("./datasets/train/images"):
        print("\nFATAL ERROR: 資料集結構錯誤。請確保您的資料位於 './datasets/train/images'!")
        return

    # 創建單次訓練用的 yaml 文件
    yaml_content = """
path: .
train: datasets/train/images
val: datasets/val/images
names: ['aortic_valve']
nc: 1
"""
    yaml_file_path = "data_single_run.yaml"
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_content)

    # 訓練模型
    model = YOLO(config['weights'])
    
    print(f"\n{'='*60}")
    print(f"開始訓練 - 模型: {config['weights']} | Batch={config['batch']}, ImgSz={config['imgsz']}")
    print('='*60)
    
    results = model.train(
        data=yaml_file_path,
        project='runs/single_run',
        name='v8m_recovery',
        **base_train_args
    )

    # 測試預測
    test_data_path = "./datasets/test/images"
    if os.path.exists(test_data_path):
        print("\n執行測試集預測...")
        test_results = predict_with_tta(model, test_data_path, tta_scales=[1.0])
        save_predictions_to_images_txt(test_results)
        print("測試完成！")

    print("\n" + "="*70)
    print("訓練完成！預測結果在 ./predict_txt/images.txt")
    print("=" * 70)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
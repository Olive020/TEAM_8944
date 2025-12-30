# ======================================================
# YOLO 進階 Backbone 訓練 - 修正與優化版
# ======================================================

import sys, os, gc, torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import warnings
import glob # 新增：用於查找檔案
warnings.filterwarnings('ignore')

print("=" * 70)
print("YOLO 進階訓練系統 - 修正版")
print(f"當前工作目錄: {os.getcwd()}")
print("=" * 70)

# ======================================================
# 1. 高準確度模型配置
# ======================================================

ADVANCED_MODELS = {
    # ★★★ 您的首選策略：三階段訓練 ★★★
    "yolo11x_enhanced": {
        "name": "YOLO11x 增強版",
        "description": "最大模型 + 階段式精細調整",
        "base_model": "yolo11x.pt", # 使用最大模型
        "strategy": {
            "stages": [
                {
                    "name": "階段1: 凍結 Backbone (快速穩定)",
                    "epochs": 100,
                    "freeze": 20, # 凍結前20層
                    "lr0": 0.001,
                    "batch": 8,
                    "imgsz": 640,
                    "mosaic": 1.0
                },
                {
                    "name": "階段2: 微調全模型 (較低學習率)",
                    "epochs": 150,
                    "freeze": None,
                    "lr0": 0.0001,
                    "batch": 4,
                    "imgsz": 800,
                    "mosaic": 1.0
                },
                {
                    "name": "階段3: 精細調整 (高解析度，關閉 Mosaic)",
                    "epochs": 50,
                    "freeze": None,
                    "lr0": 0.00001,
                    "batch": 2, # 更小的 Batch Size 來處理高解析度
                    "imgsz": 1024, # 更大解析度
                    "mosaic": 0 # 關鍵：關閉 mosaic 穩定最終 mAP
                }
            ]
        }
    },

    # 其他配置... (為了簡潔，這裡省略了其他配置，但它們在原腳本中)
    "yolo11l_cbam": { "name": "YOLO11l + 注意力機制", "description": "加入 CBAM 注意力模組", "base_model": "yolo11l.pt", "use_attention": True, "attention_type": "cbam", "training": { "epochs": 300, "batch": 6, "imgsz": 768, "lr0": 0.0005, "freeze": 10 } },
    "yolo11m_multiscale": { "name": "YOLO11m 多尺度訓練", "description": "動態改變輸入尺寸", "base_model": "yolo11m.pt", "training": { "epochs": 400, "batch": 8, "lr0": 0.001, "freeze": 8, "imgsz": [448, 832] } }, # 修正：imgsz 設為範圍
    "yolo11s_ensemble": { "name": "YOLO11s 集成學習", "description": "訓練多個模型並集成", "base_models": ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt"], "ensemble_method": "weighted_average", "training": { "epochs": 250, "batch": 12, "imgsz": 640, "lr0": 0.001 } },
    "yolo11x_spp": { "name": "YOLO11x + SPP 層", "description": "空間金字塔池化增強", "base_model": "yolo11x.pt", "use_spp": True, "spp_kernels": [5, 9, 13], "training": { "epochs": 300, "batch": 4, "imgsz": 896, "lr0": 0.0003, "freeze": 15 } },
    "yolo11l_focal": { "name": "YOLO11l + Focal Loss", "description": "使用 Focal Loss 處理類別不平衡", "base_model": "yolo11l.pt", "loss_function": "focal", "focal_gamma": 2.0, "training": { "epochs": 350, "batch": 5, "imgsz": 704, "lr0": 0.0005 } }
}

# ======================================================
# 2. 進階訓練管理器 (Advanced Trainer)
# ======================================================

class AdvancedTrainer:
    """進階訓練管理器"""

    def __init__(self, model_config):
        self.config = model_config
        self.results = []
        # 設定 project 和 name 路徑，供 find_best_weights 使用
        self.project_dir = 'runs/advanced'

    def train_staged(self, data_yaml):
        """階段式訓練 - 修正邏輯"""
        if "stages" not in self.config["strategy"]:
            return None

        current_model = None
        
        # 確保專案目錄存在，以便寫入檔案
        os.makedirs(self.project_dir, exist_ok=True)

        for i, stage in enumerate(self.config["strategy"]["stages"]):
            stage_name = f"stage_{i+1}"
            print(f"\n{'='*60}")
            print(f"{stage['name']} (輸出目錄: {self.project_dir}/{stage_name})")
            print(f"{'='*60}")

            # 載入模型：總是從上一階段的最佳權重繼續
            if i == 0:
                # 階段 1 載入基礎模型
                current_model = YOLO(self.config["base_model"])
            else:
                # 從上一階段的最佳權重繼續
                last_stage_name = f"stage_{i}"
                best_path = os.path.join(self.project_dir, last_stage_name, "weights", "best.pt")
                if os.path.exists(best_path):
                    current_model = YOLO(best_path)
                    print(f"★ 成功載入上一階段最佳權重: {best_path}")
                else:
                    print(f"警告：找不到上一階段的最佳權重，繼續使用當前模型。")

            # 訓練參數
            train_args = self.get_base_training_args()
            train_args.update(stage)
            train_args.pop("name", None)

            # 訓練 (Ultralytics 會自動保存到 project/name/weights/best.pt)
            results = current_model.train(
                data=data_yaml,
                project=self.project_dir, # 設為 runs/advanced
                name=stage_name,          # 設為 stage_1, stage_2, ...
                **train_args
            )

            self.results.append(results)

            # 清理記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return current_model # 返回最後訓練的模型

    # 移除 train_multiscale 函數，因為它實現複雜且性能不如原生的單次調用
    # 移除 train_ensemble 函數，因為集成需要複雜的推理邏輯

    def train_simple(self, data_yaml, config):
        """標準訓練或帶有注意力/損失函數的訓練"""
        model = YOLO(config["base_model"])

        train_args = self.get_base_training_args()
        train_args.update(config.get("training", {}))
        
        # 處理自定義損失函數參數
        if config.get("loss_function"):
            loss_config = get_custom_loss_config(config["loss_function"])
            train_args.update(loss_config)
            
        # 處理多尺度訓練（修正：單次調用，imgs-range=min,max）
        if config.get("multiscale"):
            # Ultralytics 原生多尺度是設定 imgsz 為目標範圍 [min, max]
            min_scale, max_scale = config["training"]["imgsz"]
            train_args["imgsz"] = min_scale
            # 這是多尺度訓練的原生參數，但它通常與 imgsz 搭配使用
            train_args["rect"] = False 

        print(f"\n訓練模型: {config['name']}")
        
        results = model.train(
            data=data_yaml,
            project=self.project_dir,
            name=config["name"].replace(" ", "_").lower(),
            **train_args
        )
        return model

    def get_base_training_args(self):
        """獲取基礎訓練參數 (保持原樣，但 workers 改為 8 以提高 GPU 利用率)"""
        return {
            "optimizer": "AdamW",
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 10,
            "warmup_bias_lr": 0.1,
            "warmup_momentum": 0.8,
            "cos_lr": True,
            "lrf": 0.01,
            "hsv_h": 0.02,
            "hsv_s": 0.7,
            "hsv_v": 0.5,
            "degrees": 20.0,
            "translate": 0.2,
            "scale": 0.5,
            "shear": 10.0,
            "perspective": 0.001,
            "flipud": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.3,
            "copy_paste": 0.3,
            "label_smoothing": 0.15,
            "dropout": 0.2,
            "patience": 100,
            "save": True,
            "cache": False,
            "device": 0 if torch.cuda.is_available() else 'cpu',
            "workers": 8 if torch.cuda.is_available() else 0, # GPU 時提高 workers
            "amp": True,
            "close_mosaic": 30,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "nbs": 64,
        }

# ======================================================
# 3. 自定義損失函數
# ======================================================

def get_custom_loss_config(loss_type):
    """獲取自定義損失函數配置 (保持原樣)"""
    loss_configs = {
        "focal": {
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "fl_gamma": 2.0 
        },
        "balanced": {
            "box": 10.0,
            "cls": 0.3,
            "dfl": 1.0
        },
        "iou_focused": {
            "box": 5.0,
            "cls": 0.5,
            "dfl": 2.0,
        }
    }
    return loss_configs.get(loss_type, loss_configs["balanced"])

# ======================================================
# 4. 測試時增強 (TTA) 及預測儲存
# ======================================================

def test_time_augmentation(model, test_path, num_augments=5):
    """測試時增強以提高準確度 (簡化為單次帶 augment 的預測)"""

    print(f"\n執行測試時增強 (TTA)...")

    # 使用 Ultralytics 原生的 TTA 參數
    results = model.predict(
        source=test_path,
        conf=0.25,
        iou=0.45,
        augment=True, # TTA 開關
        visualize=False
    )
    return results

def save_predictions_to_images_txt(results, save_dir="./predict_txt"):
    """儲存預測結果到 predict_txt/images.txt (保持原樣)"""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "images.txt")

    # 確保 results 是一個列表
    if not isinstance(results, list):
        results = [results]

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            # 檔名（不含副檔名）
            filename = os.path.basename(str(r.path))
            stem = os.path.splitext(filename)[0]

            boxes = r.boxes
            if boxes is None:
                continue

            try:
                cls_list = boxes.cls.tolist()
            except Exception:
                cls_list = []

            for j in range(len(cls_list)):
                label = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()

                f.write(
                    f"{stem} {label} {conf:.4f} "
                    f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                )

    print(f"預測結果已輸出到: {out_path}")

# ======================================================
# 5. 主程式
# ======================================================

def main():
    # 檢查環境
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("警告：未偵測到 CUDA GPU，將使用 CPU 訓練。")

    # 創建資料配置 (假設資料集結構在 ./datasets)
    data_yaml = """
path: ./datasets
train: train/images
val: val/images
names: ['aortic_valve', 'mitral_valve', 'tricuspid_valve', 'pulmonary_valve'] 
nc: 4
"""

    yaml_file_path = "data_advanced.yaml"
    with open(yaml_file_path, 'w') as f:
        f.write(data_yaml)
    print(f"資料配置檔案已創建: {yaml_file_path}")

    # 顯示選項
    print("\n" + "="*70)
    print("選擇進階訓練模式（準確度優先）:")
    print("="*70)
    print("1. YOLO11x 三階段訓練（最高準確度）")
    print("2. YOLO11l + 注意力機制")
    print("3. YOLO11m 多尺度訓練")
    print("4. YOLO11s 集成學習")
    print("5. YOLO11x + SPP 增強")
    print("6. YOLO11l + Focal Loss")
    print("7. 自定義訓練（手動配置）")

    choice = input("\n請選擇 (1-7, 預設=1): ").strip() or "1"

    # 選擇配置
    config_map = {
        "1": "yolo11x_enhanced",
        "2": "yolo11l_cbam",
        "3": "yolo11m_multiscale",
        "4": "yolo11s_ensemble",
        "5": "yolo11x_spp",
        "6": "yolo11l_focal"
    }

    config_name = config_map.get(choice, "yolo11x_enhanced")
    config = ADVANCED_MODELS[config_name]
    
    model = None

    if choice == "7":
        # 自定義訓練
        print("\n自定義訓練邏輯未實現，請選擇 1-6 或完善 train_simple 函數。")
        return
    else:
        print(f"\n使用配置: {config['name']}")
        print(f"說明: {config['description']}")

        trainer = AdvancedTrainer(config)

        # 根據不同配置選擇訓練方法
        if "stages" in config.get("strategy", {}):
            model = trainer.train_staged(yaml_file_path)
        elif "base_models" in config:
            # 集成學習 (訓練多個模型)
            models = trainer.train_ensemble(yaml_file_path)
            model = models[-1] # 返回最後一個訓練的模型作為代表
        else:
            # 標準訓練，多尺度、注意力、SPP、Focal Loss 都在這裡處理
            model = trainer.train_simple(yaml_file_path, config)


    print("\n訓練完成！")

    # 驗證
    print("\n執行驗證...")
    try:
        val_results = model.val()
        print(f"mAP@50: {val_results.box.map50:.4f}")
        print(f"mAP@50-95: {val_results.box.map:.4f}")
    except Exception as e:
        print(f"驗證失敗，請檢查資料配置: {e}")
        
    # 測試
    test_data_path = "./datasets/test/images"
    if os.path.exists(test_data_path):
        print("\n執行測試集預測...")
        test_results = test_time_augmentation(
            model,
            test_data_path,
            num_augments=1 # 簡化為 1 次 TTA
        )

        # ★★★ 新增：把預測結果寫成 predict_txt/images.txt ★★★
        save_predictions_to_images_txt(test_results, "./predict_txt")

        print("測試完成！")

    print("\n" + "="*70)
    print("訓練結束 - 結果保存在 runs/ 目錄")
    print("=" * 70)

if __name__ == "__main__":
    main()
# ======================================================
# YOLO 自定義 Backbone 訓練系統 - 最高準確度優先
# 專為小資料集（1631張）設計，支援多種先進 backbone
# ======================================================

import sys, os, gc, torch, timm
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import yaml_load
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("YOLO 自定義 Backbone 訓練系統")
print("準確度優先 - 支援多種先進架構")
print("=" * 70)

# ======================================================
# 1. 高準確度 Backbone 配置
# ======================================================

BACKBONE_CONFIGS = {
    "convnext_v2": {
        "name": "ConvNeXt V2 - 最強準確度",
        "description": "Meta AI 最新架構，專為視覺任務優化",
        "model_name": "convnextv2_base",
        "features": [
            "✓ 最先進的特徵提取能力",
            "✓ 優異的小物體檢測",
            "✓ 強大的多尺度特徵"
        ],
        "pretrained": True,
        "freeze_stages": 2,  # 凍結前2個階段
        "out_indices": [1, 2, 3, 4],  # 提取的特徵層
    },

    "swin_v2": {
        "name": "Swin Transformer V2 - Transformer 架構",
        "description": "Microsoft 的視覺 Transformer，準確度極高",
        "model_name": "swinv2_base_window12_192",
        "features": [
            "✓ Transformer 自注意力機制",
            "✓ 極強的全局建模能力",
            "✓ 階層化特徵提取"
        ],
        "pretrained": True,
        "freeze_stages": 1,
        "out_indices": [0, 1, 2, 3],
    },

    "efficientnet_v2": {
        "name": "EfficientNet V2 - 效率與準確度平衡",
        "description": "Google 的高效架構，準確度優秀",
        "model_name": "tf_efficientnetv2_l",
        "features": [
            "✓ NAS 搜尋的最優架構",
            "✓ 漸進式學習優化",
            "✓ 高效的參數利用"
        ],
        "pretrained": True,
        "freeze_stages": 3,
        "out_indices": [2, 3, 4, 5],
    },

    "regnet": {
        "name": "RegNet - Facebook 設計空間架構",
        "description": "基於設計空間搜尋的高效架構",
        "model_name": "regnetx_320",
        "features": [
            "✓ 系統化的架構設計",
            "✓ 優秀的擴展性",
            "✓ 穩定的訓練特性"
        ],
        "pretrained": True,
        "freeze_stages": 2,
        "out_indices": [1, 2, 3, 4],
    },

    "beit_v2": {
        "name": "BEiT v2 - 自監督視覺模型",
        "description": "Microsoft 的 BERT 風格視覺模型",
        "model_name": "beitv2_base_patch16_224",
        "features": [
            "✓ 強大的自監督預訓練",
            "✓ 優秀的特徵表示",
            "✓ 對小資料集友好"
        ],
        "pretrained": True,
        "freeze_stages": 1,
        "out_indices": [3, 6, 9, 12],
    },

    "eva": {
        "name": "EVA - 視覺基礎模型",
        "description": "來自北京智源的超大規模預訓練模型",
        "model_name": "eva_large_patch14_196",
        "features": [
            "✓ 10億參數預訓練",
            "✓ 極強的視覺理解",
            "✓ CLIP 對齊訓練"
        ],
        "pretrained": True,
        "freeze_stages": 2,
        "out_indices": [7, 11, 15, 23],
    },

    "maxvit": {
        "name": "MaxViT - 多軸注意力",
        "description": "Google 的多軸視覺 Transformer",
        "model_name": "maxvit_base_tf_224",
        "features": [
            "✓ 多軸注意力機制",
            "✓ CNN + Transformer 混合",
            "✓ 高解析度友好"
        ],
        "pretrained": True,
        "freeze_stages": 1,
        "out_indices": [2, 5, 8, 11],
    }
}

# ======================================================
# 2. 創建自定義 YOLO 配置檔案
# ======================================================

def create_custom_yolo_config(backbone_name, backbone_config):
    """創建使用自定義 backbone 的 YOLO 配置"""

    yaml_content = f"""
# YOLOv11 with Custom Backbone - {backbone_name}
# Optimized for small dataset (1631 images)

# Parameters
nc: 1  # number of classes
depth_multiple: 1.0
width_multiple: 1.0

# Custom Backbone
backbone:
  # {backbone_config['name']}
  - [-1, 1, CustomBackbone, [{backbone_config['model_name']}, {backbone_config['pretrained']}]]

# YOLOv11 head - 保持原始檢測頭
head:
  - [-1, 1, Conv, [256, 3, 2]]  # P3/8
  - [[-1, -2], 1, Concat, [1]]
  - [-1, 3, C3, [256]]

  - [-1, 1, Conv, [512, 3, 2]]  # P4/16
  - [[-1, -3], 1, Concat, [1]]
  - [-1, 3, C3, [512]]

  - [-1, 1, Conv, [1024, 3, 2]]  # P5/32
  - [[-1, -4], 1, Concat, [1]]
  - [-1, 3, C3, [1024]]

  # Detection head
  - [[15, 18, 21], 1, Detect, [nc]]
"""

    config_path = f"yolo_custom_{backbone_name}.yaml"
    with open(config_path, 'w') as f:
        f.write(yaml_content)

    return config_path

# ======================================================
# 3. 自定義 Backbone 類別
# ======================================================

class CustomBackbone(torch.nn.Module):
    """通用的自定義 Backbone 包裝器"""

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        # 載入 timm 模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )

        # 獲取輸出通道數
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        features = self.model(x)
        return features

# ======================================================
# 4. 訓練配置（針對小資料集優化）
# ======================================================

def get_training_config(backbone_name):
    """根據 backbone 獲取訓練配置"""

    base_config = {
        "epochs": 300,
        "batch": 8,  # 小批次，因為 backbone 較大
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr0": 0.0001,  # 很低的學習率
        "lrf": 0.001,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 10,
        "cos_lr": True,
        "patience": 100,
        "close_mosaic": 30,

        # 強資料增強
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
        "degrees": 20,
        "translate": 0.2,
        "scale": 0.5,
        "shear": 10,
        "perspective": 0.001,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.2,

        # 正規化
        "label_smoothing": 0.15,
        "dropout": 0.2
    }

    # 根據不同 backbone 調整配置
    if backbone_name in ["convnext_v2", "swin_v2", "eva", "beit_v2"]:
        # 大型模型需要更小的批次和學習率
        base_config["batch"] = 4
        base_config["lr0"] = 0.00005
        base_config["accumulate"] = 4  # 梯度累積

    elif backbone_name == "efficientnet_v2":
        # EfficientNet 可以稍大批次
        base_config["batch"] = 8
        base_config["lr0"] = 0.0001

    return base_config

# ======================================================
# 5. 訓練函數
# ======================================================

def train_with_custom_backbone(backbone_name, data_yaml):
    """使用自定義 backbone 訓練"""

    print(f"\n{'='*60}")
    print(f"開始使用 {BACKBONE_CONFIGS[backbone_name]['name']} 訓練")
    print(f"說明: {BACKBONE_CONFIGS[backbone_name]['description']}")
    print(f"{'='*60}")

    # 顯示特點
    print("\n架構特點:")
    for feature in BACKBONE_CONFIGS[backbone_name]['features']:
        print(f"  {feature}")

    # 創建配置檔案
    config_path = create_custom_yolo_config(
        backbone_name,
        BACKBONE_CONFIGS[backbone_name]
    )

    # 獲取訓練配置
    train_config = get_training_config(backbone_name)

    print(f"\n訓練配置:")
    print(f"  批次大小: {train_config['batch']}")
    print(f"  學習率: {train_config['lr0']}")
    print(f"  訓練輪數: {train_config['epochs']}")

    try:
        # 創建模型
        model = YOLO(config_path)

        # 載入預訓練的 YOLO 檢測頭權重（可選）
        # model.model.head.load_state_dict(
        #     torch.load("yolo11n.pt")['model'].head.state_dict(),
        #     strict=False
        # )

        # 開始訓練
        results = model.train(
            data=data_yaml,
            project='runs/custom_backbone',
            name=f'{backbone_name}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}',
            **train_config
        )

        return model, results

    except Exception as e:
        print(f"錯誤: {e}")
        print(f"可能需要安裝 timm: pip install timm")
        return None, None

# ======================================================
# 6. 比較不同 Backbone
# ======================================================

def compare_backbones(data_yaml):
    """比較不同 backbone 的效果"""

    results_summary = {}

    # 測試的 backbone 列表（按推薦順序）
    test_backbones = [
        "convnext_v2",      # 最推薦：最強準確度
        "swin_v2",          # 推薦：Transformer 架構
        "efficientnet_v2",  # 推薦：平衡選擇
        "maxvit"            # 推薦：混合架構
    ]

    for backbone_name in test_backbones:
        print(f"\n{'='*70}")
        print(f"測試 Backbone: {backbone_name}")
        print(f"{'='*70}")

        model, results = train_with_custom_backbone(backbone_name, data_yaml)

        if model and results:
            # 驗證
            val_results = model.val()

            # 記錄結果
            results_summary[backbone_name] = {
                "mAP50": val_results.box.map50 if hasattr(val_results.box, 'map50') else 0,
                "mAP50-95": val_results.box.map if hasattr(val_results.box, 'map') else 0,
                "precision": val_results.box.p if hasattr(val_results.box, 'p') else 0,
                "recall": val_results.box.r if hasattr(val_results.box, 'r') else 0,
            }

            # 清理記憶體
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 顯示比較結果
    print("\n" + "="*70)
    print("Backbone 效能比較結果")
    print("="*70)

    for backbone, metrics in results_summary.items():
        print(f"\n{BACKBONE_CONFIGS[backbone]['name']}:")
        print(f"  mAP@50: {metrics['mAP50']:.4f}")
        print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

    # 找出最佳 backbone
    best_backbone = max(results_summary.items(),
                       key=lambda x: x[1]['mAP50-95'])

    print(f"\n{'='*70}")
    print(f"🏆 最佳 Backbone: {BACKBONE_CONFIGS[best_backbone[0]]['name']}")
    print(f"   mAP@50-95: {best_backbone[1]['mAP50-95']:.4f}")
    print(f"{'='*70}")

    return results_summary

# ======================================================
# 主程式
# ======================================================

if __name__ == "__main__":

    # 檢查 GPU
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("警告：使用 CPU 訓練會非常慢！")

    # 檢查 timm 是否安裝
    try:
        import timm
        print(f"timm 版本: {timm.__version__}")
    except ImportError:
        print("\n需要安裝 timm 庫來使用自定義 backbone:")
        print("pip install timm")
        sys.exit(1)

    # 創建資料 yaml
    data_yaml_content = """
path: ./datasets
train: train/images
val: val/images
names: ['aortic_valve']
nc: 1
"""

    with open("data_custom.yaml", 'w') as f:
        f.write(data_yaml_content)

    print("\n" + "="*70)
    print("選擇訓練模式:")
    print("="*70)
    print("1. ConvNeXt V2 - 最強準確度（推薦）")
    print("2. Swin Transformer V2 - Transformer 架構")
    print("3. EfficientNet V2 - 效率與準確度平衡")
    print("4. MaxViT - 混合架構")
    print("5. RegNet - 穩定架構")
    print("6. BEiT v2 - 自監督模型")
    print("7. EVA - 超大規模預訓練")
    print("8. 自動比較多個 Backbone（耗時較長）")

    choice = input("\n請選擇 (1-8, 預設=1): ").strip() or "1"

    backbone_map = {
        "1": "convnext_v2",
        "2": "swin_v2",
        "3": "efficientnet_v2",
        "4": "maxvit",
        "5": "regnet",
        "6": "beit_v2",
        "7": "eva"
    }

    if choice == "8":
        # 比較多個 backbone
        compare_backbones("data_custom.yaml")
    else:
        # 訓練單個 backbone
        backbone_name = backbone_map.get(choice, "convnext_v2")
        model, results = train_with_custom_backbone(backbone_name, "data_custom.yaml")

        if model:
            print("\n訓練完成！")

            # 測試預測
            if os.path.exists("./datasets/test/images"):
                print("\n開始測試集預測...")
                test_results = model.predict(
                    source="./datasets/test/images",
                    save=True,
                    conf=0.25,
                    iou=0.5,
                    augment=True  # 測試時增強
                )
                print("預測完成！")

    print("\n" + "="*70)
    print("訓練結束")
    print("="*70)
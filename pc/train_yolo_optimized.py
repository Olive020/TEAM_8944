# ======================================================
# YOLO11 主動脈瓣偵測優化訓練程式（使用現有資料夾 + 防 OOM + 兼容舊版 CUDA）
# ======================================================

import sys, pkgutil, locale, os, zipfile, shutil, gc, multiprocessing, glob, json
from datetime import datetime
from ultralytics import YOLO

# ---------------------------
# 使用現有資料集（不要搬動檔案）
# ---------------------------
USE_EXISTING_DATASET = True

# 你現有資料夾（請確認 val 也有對應資料夾；若沒有，把 VAL_* 指到你的驗證集）
TRAIN_IMAGES = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\images"
TRAIN_LABELS = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\labels"
VAL_IMAGES   = r"C:\Users\Lab902\Desktop\week8\pc\datasets\val\images"
VAL_LABELS   = r"C:\Users\Lab902\Desktop\week8\pc\datasets\val\labels"

# 動態產生的 YAML 路徑
AUTO_DATA_YAML = "./_auto_aortic_valve.yaml"

# ---------------------------
# 系統環境檢查
# ---------------------------
print("Python 路徑:", sys.executable)
print("ipykernel 模組可用:", pkgutil.find_loader("ipykernel") is not None)

# 強制 UTF-8
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# ---- 安全地設定 CUDA 記憶體配置（舊版 PyTorch 會自動退避） ----
def _safe_set_cuda_alloc_conf():
    val = "max_split_size_mb:128"  # 注意是冒號
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = val
    try:
        import torch
        if torch.cuda.is_available():
            _ = torch.cuda.device_count()
    except Exception as e:
        if "Unrecognized CachingAllocator option" in str(e):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

_safe_set_cuda_alloc_conf()

# ======================================================
# 模型配置定義
# ======================================================

MODEL_CONFIGS = {
    "high_accuracy": {
        "name": "高準確度模式",
        "description": "YOLO11x",
        "weights": "yolo11x.pt",
        "epochs": 75,
        "batch": 16,
        "imgsz": 800,  # 實際會限縮到 <=640
        "patience": 60,
        "lr0": 0.0008,
        "optimizer": "AdamW",
        "augmentation": "strong"
    },
    "balanced": {
        "name": "平衡模式",
        "description": "YOLO11l",
        "weights": "yolo11l.pt",
        "epochs": 100,
        "batch": 24,
        "imgsz": 640,
        "patience": 50,
        "lr0": 0.001,
        "optimizer": "AdamW",
        "augmentation": "medium"
    },
    "fast_training": {
        "name": "快速訓練模式",
        "description": "YOLO11m",
        "weights": "yolo11m.pt",
        "epochs": 100,
        "batch": 32,
        "imgsz": 640,
        "patience": 40,
        "lr0": 0.0015,
        "optimizer": "SGD",
        "augmentation": "light"
    },
    "ultra_fast": {
        "name": "超快速模式",
        "description": "YOLO11n",
        "weights": "yolo11n.pt",
        "epochs": 100,
        "batch": 64,
        "imgsz": 640,
        "patience": 30,
        "lr0": 0.002,
        "optimizer": "SGD",
        "augmentation": "minimal"
    },
    
    "ultra_fast_plus" : {
    "name": "超快速模式（醫療版）",
    "description": "YOLO11n，小模型 + 醫療增強",
    "weights": "yolo11n.pt",
    "epochs": 150,          # 從 100 拉到 150
    "batch": -1,            # 讓 Ultralytics 自動找安全 batch
    "imgsz": 640,           # 顯存不夠就維持 640
    "patience": 60,
    "lr0": 0.0010,          # 稍微保守一點
    "optimizer": "AdamW",   # 改成 AdamW 比 SGD 穩
    "augmentation": "medical"
}


}

# ======================================================
# 數據增強配置
# ======================================================

AUGMENTATION_CONFIGS = {
    "strong":  {"hsv_h":0.015,"hsv_s":0.4,"hsv_v":0.4,"degrees":10.0,"translate":0.15,"scale":0.5,"shear":5.0,"flipud":0.5,"fliplr":0.5,"mosaic":1.0,"mixup":0.3,"copy_paste":0.1},
    "medium":  {"hsv_h":0.01,"hsv_s":0.3,"hsv_v":0.3,"degrees":5.0,"translate":0.1,"scale":0.3,"shear":2.0,"flipud":0.5,"fliplr":0.5,"mosaic":0.8,"mixup":0.15,"copy_paste":0.0},
    "light":   {"hsv_h":0.005,"hsv_s":0.2,"hsv_v":0.2,"degrees":3.0,"translate":0.05,"scale":0.2,"shear":1.0,"flipud":0.3,"fliplr":0.3,"mosaic":0.5,"mixup":0.1,"copy_paste":0.0},
    "minimal": {"hsv_h":0.0,"hsv_s":0.1,"hsv_v":0.1,"degrees":0.0,"translate":0.0,"scale":0.1,"shear":0.0,"flipud":0.2,"fliplr":0.2,"mosaic":0.3,"mixup":0.0,"copy_paste":0.0}

}
# ======================================================
# 醫療影像專用增強（CT、灰階影像）
# ======================================================
AUGMENTATION_CONFIGS["medical"] = {
    # 灰階 CT：不改色相、不改飽和度，只做亮度
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.25,     # 亮度 ±25%

    # 幾何：小幅度即可，避免失真
    "degrees": 3.0,    # 小角度
    "translate": 0.05, # 平移 5%
    "scale": 0.15,     # 縮放 15%
    "shear": 0.0,

    # 翻轉：上下翻對心臟不合理，關；左右翻可小幅開
    "flipud": 0.0,
    "fliplr": 0.2,

    # 醫療影像不建議的合成類增強全部關閉
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0
}


# ======================================================
# 工具函數
# ======================================================

def assert_dir(p, name):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"[資料夾不存在] {name}: {p}")

def write_data_yaml(train_images, val_images, names=("aortic_valve",)):
    """
    產生給 Ultralytics 使用的 data.yaml
    - 單一類別：names: ['aortic_valve']（你也可以改）
    - Ultralytics 只要 train/val 指到 images 目錄；labels 依規則自動對應
    """
    nc = len(names)
    content = (
        f"path: .\n"
        f"train: {train_images}\n"
        f"val: {val_images}\n"
        f"names: {list(names)}\n"
        f"nc: {nc}\n"
    )
    with open(AUTO_DATA_YAML, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] 已寫出 data.yaml → {AUTO_DATA_YAML}")
    return AUTO_DATA_YAML

def unzip_if_needed(zip_path, dest_dir):
    if os.path.isdir(dest_dir):
        return
    if os.path.exists(zip_path):
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

def find_patient_root(root):
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.lower().startswith("patient") for d in dirnames):
            return dirpath
    return root

def clean_up(vars_to_delete=None):
    if vars_to_delete:
        for v in vars_to_delete:
            try: del v
            except Exception: pass
    gc.collect()
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
    except Exception:
        pass

def auto_select_device():
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"

def print_available_configs():
    print("\n" + "=" * 60)
    print("可用的訓練配置:")
    print("=" * 60)
    for key, config in MODEL_CONFIGS.items():
        print(f"\n[{key}]")
        print(f"  名稱: {config['name']}")
        print(f"  描述: {config['description']}")
        print(f"  模型: {config['weights']}")
        print(f"  訓練輪數: {config['epochs']}")
        print(f"  批次大小: {config['batch']}")
        print(f"  圖片尺寸: {config['imgsz']}")
    print("\n" + "=" * 60)

# ======================================================
# 訓練 / 評估 / 預測
# ======================================================

def train_model_optimized(config_name="balanced", device_choice="cpu", custom_params=None, data_yaml_path=AUTO_DATA_YAML):
    if config_name not in MODEL_CONFIGS:
        print(f"錯誤: 未知配置 {config_name}")
        return None, None

    config = dict(MODEL_CONFIGS[config_name])
    aug_config = AUGMENTATION_CONFIGS[config["augmentation"]]
    if custom_params: config.update(custom_params)

    print("=" * 60)
    print(f"開始訓練 - {config['name']}")
    print(f"描述: {config['description']}")
    print(f"模型: {config['weights']}, Epochs: {config['epochs']}, Batch: {config['batch']}")
    print("=" * 60)

    model = YOLO(config['weights'])

    train_params = {
        'data': data_yaml_path,                 # ← 使用動態 YAML
        'epochs': config['epochs'],
        'batch': -1,
        'imgsz': min(config['imgsz'], 640),
        'device': device_choice,
        'optimizer': config['optimizer'],
        'lr0': config['lr0'],
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'patience': config['patience'],
        'save_period': -1,
        'cache': 'disk',
        'workers': 2,
        'project': 'runs/detect',
        'name': f'train_{config_name}_{datetime.now().strftime("%Y%m%d_%H%M")}',
        'exist_ok': False,
        'pretrained': True,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'val': True,
        'plots': False,
        'save': True,
        'single_cls': True,
        'amp': True,
        'verbose': True,
        'seed': 42,
        'close_mosaic': 10,
        **aug_config
    }

    def _do_train(): return model.train(**train_params)

    try:
        results = _do_train()
    except RuntimeError as e:
        s = str(e)
        if "Unrecognized CachingAllocator option" in s:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            print("[警告] 偵測到不支援的 PYTORCH_CUDA_ALLOC_CONF，已移除並重試訓練。")
            results = _do_train()
        elif "CUDA" in s or "cuda" in s:
            if train_params.get("device", None) != "cpu":
                print("[警告] CUDA 初始化失敗，改用 CPU 重試一次…")
                train_params["device"] = "cpu"
                results = _do_train()
            else:
                raise
        else:
            raise

    config_save_path = f"runs/detect/{train_params['name']}/config.json"
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config_name': config_name,
            'model_config': config,
            'augmentation_config': aug_config,
            'train_params': {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                             for k, v in train_params.items()}
        }, f, indent=2, ensure_ascii=False)

    print(f"訓練配置已保存至: {config_save_path}")
    return results, train_params['name']

def evaluate_model(model, device_choice="cpu", data_yaml_path=AUTO_DATA_YAML):
    results = model.val(
        data=data_yaml_path,
        imgsz=640,
        batch=8,
        device=device_choice,
        save_json=False,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=False,
        dnn=False,
        plots=False,
        verbose=True
    )

    print("=" * 60)
    print("評估結果:")
    try:
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
    except Exception:
        print("（注意：評估回傳物件結構可能隨版本不同，請以實際輸出為準）")
    print("=" * 60)
    return results

def prepare_testset_and_predict(model, device_choice="cpu", imgsz=640):
    """
    測試資料預測：若你也有固定的 test 目錄，可直接改成你的 test 路徑並移除複製流程。
    目前保留原本行為：把 testing_image.zip 解壓後複製到 ./datasets/test/images。
    """
    unzip_if_needed("testing_image.zip", "./testing_image")
    TEST_ROOT = find_patient_root("./testing_image")

    DST_TEST = "./datasets/test/images"
    os.makedirs(DST_TEST, exist_ok=True)

    all_files = []
    if os.path.isdir(TEST_ROOT):
        for patient_folder in os.listdir(TEST_ROOT):
            patient_path = os.path.join(TEST_ROOT, patient_folder)
            if os.path.isdir(patient_path) and patient_folder.lower().startswith("patient"):
                for fname in os.listdir(patient_path):
                    if fname.lower().endswith(".png"):
                        all_files.append(os.path.join(patient_path, fname))
    all_files.sort()

    copied = 0
    for f in all_files:
        dst = os.path.join(DST_TEST, os.path.basename(f))
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(f, dst)
        copied += 1

    print(f"測試集準備完成：{copied} 張圖片")

    results = model.predict(
        source=DST_TEST,
        save=False,
        imgsz=imgsz,
        device=device_choice,
        conf=0.25,
        iou=0.45,
        max_det=300,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        verbose=True
    )

    print(f"預測完成：共 {len(results)} 張圖片")

    os.makedirs('./predict_txt', exist_ok=True)
    with open('./predict_txt/images.txt', 'w', encoding='utf-8') as output_file:
        for i in range(len(results)):
            filename = str(results[i].path).replace('\\', '/').split('/')[-1].split('.png')[0]
            boxes = results[i].boxes
            try:
                cls_list = boxes.cls.tolist()
            except Exception:
                cls_list = []
            for j in range(len(cls_list)):
                label = int(boxes.cls[j].item())
                conf = boxes.conf[j].item()
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                output_file.write(f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")

    return results, all_files

# ======================================================
# main
# ======================================================

def main():
    device_choice = auto_select_device()
    print(f"自動選擇裝置: {device_choice}")

    try:
        import torch
        if device_choice != "cpu" and not torch.cuda.is_available():
            device_choice = "cpu"
            print("[提示] 偵測到 CUDA 不可用，已改用 CPU。")
    except Exception:
        device_choice = "cpu"
        print("[提示] PyTorch CUDA 探測異常，已改用 CPU。")

    print_available_configs()

    # 選擇配置
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "high_accuracy"
        print(f"\n使用預設配置: {config_name}")
        print("提示: 可透過命令列參數指定配置，例如: python train_yolo_optimized.py balanced")

    # === 建立 data.yaml（不搬動你的檔案） ===
    if USE_EXISTING_DATASET:
        assert_dir(TRAIN_IMAGES, "TRAIN_IMAGES")
        assert_dir(TRAIN_LABELS, "TRAIN_LABELS")
        assert_dir(VAL_IMAGES,   "VAL_IMAGES")
        assert_dir(VAL_LABELS,   "VAL_LABELS")
        data_yaml_path = write_data_yaml(TRAIN_IMAGES, VAL_IMAGES, names=("aortic_valve",))
    else:
        # 若你之後想用 zip 自動拆分，才走舊流程
        raise RuntimeError("目前設定 USE_EXISTING_DATASET=True，如要用 zip 拆分請改成 False。")

    # === 訓練 ===
    train_results, run_name = train_model_optimized(
        config_name=config_name,
        device_choice=device_choice,
        data_yaml_path=data_yaml_path
    )

    clean_up([train_results])

    # === 尋找最佳權重 ===
    weight_patterns = [
        f'./runs/detect/{run_name}/weights/best.pt',
        f'./runs/detect/train_{config_name}_*/weights/best.pt',
        './runs/detect/train*/weights/best.pt'
    ]
    weights_path = None
    for pattern in weight_patterns:
        matches = glob.glob(pattern)
        if matches:
            weights_path = max(matches, key=os.path.getctime)
            break
    if not weights_path or not os.path.exists(weights_path):
        print("警告: 找不到訓練權重")
        return
    print(f"使用權重: {weights_path}")

    # === 以最佳權重建立新 model（乾淨顯存） ===
    model = YOLO(weights_path)

    # === 評估 ===
    eval_results = evaluate_model(model, device_choice, data_yaml_path)
    clean_up([eval_results])

    # === 測試預測（若你也有固定 test 目錄，這裡可自行改 source）===
    infer_imgsz = min(MODEL_CONFIGS[config_name]["imgsz"], 640)
    test_results, all_files = prepare_testset_and_predict(
        model=model,
        device_choice=device_choice,
        imgsz=infer_imgsz
    )

    clean_up([test_results, all_files, model])
    print("\n✅ 訓練流程完成！（使用現有資料夾 / 防 OOM 版）")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

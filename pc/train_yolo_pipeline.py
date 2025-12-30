# ======================================================
# YOLOv8 自動訓練與推論流程（Windows-safe）
# 作者：整理自使用者提供程式 — 由 ChatGPT 轉換
# ======================================================

import sys, pkgutil, locale, os, zipfile, shutil, gc, multiprocessing
from ultralytics import YOLO

# ---------------------------
# 系統環境檢查
# ---------------------------
print("Python 路徑:", sys.executable)
print("ipykernel 模組可用:", pkgutil.find_loader("ipykernel") is not None)

# 強制 UTF-8
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

def unzip_if_needed(zip_path, dest_dir):
    """若尚未解壓縮則解壓 zip 檔"""
    if os.path.isdir(dest_dir):
        return
    if os.path.exists(zip_path):
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

def find_patient_root(root):
    """尋找包含 patient 資料夾的根路徑"""
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.lower().startswith("patient") for d in dirnames):
            return dirpath
    return root

def move_patients(IMG_ROOT, LBL_ROOT, start, end, split):
    """將病人資料移動到指定的資料夾（train / val）"""
    moved = 0
    for i in range(start, end + 1):
        patient = f"patient{i:04d}"
        img_dir = os.path.join(IMG_ROOT, patient)
        lbl_dir = os.path.join(LBL_ROOT, patient)
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, base + ".png")
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(img_path):
                continue
            dst_img = f"./datasets/{split}/images/{base}.png"
            dst_lbl = f"./datasets/{split}/labels/{base}.txt"
            if os.path.exists(dst_img):
                os.remove(dst_img)
            if os.path.exists(dst_lbl):
                os.remove(dst_lbl)
            shutil.move(img_path, dst_img)
            shutil.move(lbl_path, dst_lbl)
            moved += 1
    return moved

def prepare_datasets():
    # 解壓訓練資料（如果有提供 zip）
    unzip_if_needed("training_image.zip", "./training_image")
    unzip_if_needed("training_label.zip", "./training_label")

    IMG_ROOT = find_patient_root("./training_image")
    LBL_ROOT = find_patient_root("./training_label")

    # 建立訓練與驗證資料夾結構
    for p in ["./datasets/train/images", "./datasets/train/labels",
              "./datasets/val/images", "./datasets/val/labels"]:
        os.makedirs(p, exist_ok=True)

    # 分割資料（預設：patient0001-0030 train，0031-0050 val）
    n_train = move_patients(IMG_ROOT, LBL_ROOT, 1, 30, "train")
    n_val   = move_patients(IMG_ROOT, LBL_ROOT, 31, 50, "val")
    print(f"完成移動：train {n_train} 筆，val {n_val} 筆")

    print('訓練集圖片數量 : ', len(os.listdir("./datasets/train/images")))
    print('訓練集標記數量 : ', len(os.listdir("./datasets/train/labels")))
    print('驗證集圖片數量 : ', len(os.listdir("./datasets/val/images")))
    print('驗證集標記數量 : ', len(os.listdir("./datasets/val/labels")))

def train_model(device_choice="cpu", epochs=50, batch=32, imgsz=640, weights='yolo12m.pt', data_yaml='./aortic_valve_colab.yaml'):
    print(f"開始訓練：device={device_choice}, epochs={epochs}, batch={batch}, imgsz={imgsz}")
    model = YOLO(weights)  # 可指定預訓練權重或自定義模型檔
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device_choice
    )
    return results

def prepare_testset_and_predict(device_choice="cpu", weights_path='./runs/detect/train/weights/best.pt', imgsz=640):
    # 解壓測試資料（如果有提供 zip）
    unzip_if_needed("testing_image.zip", "./testing_image")
    TEST_ROOT = find_patient_root("./testing_image")

    DST_TEST = "./datasets/test/images"
    os.makedirs(DST_TEST, exist_ok=True)

    # 收集所有 patient 圖片
    all_files = []
    if os.path.isdir(TEST_ROOT):
        for patient_folder in os.listdir(TEST_ROOT):
            patient_path = os.path.join(TEST_ROOT, patient_folder)
            if os.path.isdir(patient_path) and patient_folder.lower().startswith("patient"):
                for fname in os.listdir(patient_path):
                    if fname.lower().endswith(".png"):
                        all_files.append(os.path.join(patient_path, fname))
    all_files.sort()

    # 複製到 test/images
    copied = 0
    for f in all_files:
        dst = os.path.join(DST_TEST, os.path.basename(f))
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(f, dst)
        copied += 1

    print(f"來源根目錄：{TEST_ROOT}")
    print(f"完成複製！總共 {copied} 張圖片。")
    print('測試集圖片數量 : ', len(os.listdir(DST_TEST)))

    # 推論（Predict）
    model = YOLO(weights_path)
    results = model.predict(
        source=DST_TEST,
        save=True,
        imgsz=imgsz,
        device=device_choice
    )

    print(f"共偵測 {len(results)} 張圖片。`(len(results) may be 0 if no images)`")
    # 小心存取索引，避免越界
    if len(results) > 260:
        try:
            print('第 260 張的預測類別 : ', results[260].boxes.cls[0].item())
            print('預測信心分數 : ', results[260].boxes.conf[0].item())
            print('預測框座標 : ', results[260].boxes.xyxy[0].tolist())
        except Exception as e:
            print('讀取第260張預測資訊失敗：', e)

    # 儲存預測結果到文字檔
    os.makedirs('./predict_txt', exist_ok=True)
    with open('./predict_txt/images.txt', 'w', encoding='utf-8') as output_file:
        for i in range(len(results)):
            filename = str(results[i].path).replace('\\', '/').split('/')[-1].split('.png')[0]
            boxes = results[i].boxes
            try:
                cls_list = boxes.cls.tolist()
            except Exception:
                cls_list = []
            box_num = len(cls_list)
            if box_num > 0:
                for j in range(box_num):
                    label = int(boxes.cls[j].item())
                    conf = boxes.conf[j].item()
                    x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                    line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    output_file.write(line)

    return results, all_files

def clean_up(vars_to_delete=None):
    import torch as _torch
    if vars_to_delete:
        for v in vars_to_delete:
            try:
                del v
            except Exception:
                pass
    gc.collect()
    try:
        _torch.cuda.empty_cache()
    except Exception:
        pass

def auto_select_device():
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return 0  # use GPU 0
    except Exception:
        pass
    return "cpu"

def main():
    # 建議：在本函式前準備好 training_image.zip / training_label.zip / testing_image.zip / aortic_valve_colab.yaml / yolo12n.pt 等檔案
    device_choice = auto_select_device()
    print(f"自動選擇裝置: {device_choice}")

    prepare_datasets()

    # 若要用 CPU 可傳 device_choice='cpu'，若要強制 GPU 傳 '0' 或 '0,1'
    train_results = train_model(device_choice=device_choice, epochs=50, batch=32, imgsz=640)

    # 請確認 weights 路徑是否正確（通常會在 runs/detect/train/weights/best.pt）
    weights_path = './runs/detect/train/weights/best.pt'
    if not os.path.exists(weights_path):
        print(f"警告: 找不到 {weights_path}。請確認訓練後的 best.pt 路徑，或手動指定 weights_path。繼續使用預設模型進行推論可能會失敗。")


    results, all_files = prepare_testset_and_predict(device_choice=device_choice, weights_path=weights_path, imgsz=640)

    # 清理
    clean_up([results, all_files])
    print("✅ 程式執行完畢。")


if __name__ == '__main__':
    # Windows 平台 multiprocess 啟動保護
    multiprocessing.freeze_support()
    main()

# ======================================================
# YOLOv8 自動訓練與推論流程
# 作者：ChatGPT（整理自你的 train.ipynb）
# ======================================================

import sys, pkgutil, locale, os, zipfile, shutil, gc
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

# 測試 Torch 與 GPU
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 裝置:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch import error:", e)

# ---------------------------
# 資料前處理
# ---------------------------

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

# 解壓訓練資料
unzip_if_needed("training_image.zip", "./training_image")
unzip_if_needed("training_label.zip", "./training_label")

IMG_ROOT = find_patient_root("./training_image")
LBL_ROOT = find_patient_root("./training_label")

# 建立訓練與驗證資料夾結構
for p in ["./datasets/train/images", "./datasets/train/labels",
          "./datasets/val/images", "./datasets/val/labels"]:
    os.makedirs(p, exist_ok=True)

def move_patients(start, end, split):
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
            if os.path.exists(dst_img): os.remove(dst_img)
            if os.path.exists(dst_lbl): os.remove(dst_lbl)
            shutil.move(img_path, dst_img)
            shutil.move(lbl_path, dst_lbl)
            moved += 1
    return moved

# 分割資料
n_train = move_patients(1, 30, "train")
n_val   = move_patients(31, 50, "val")
print(f"完成移動：train {n_train} 筆，val {n_val} 筆")

print('訓練集圖片數量 : ', len(os.listdir("./datasets/train/images")))
print('訓練集標記數量 : ', len(os.listdir("./datasets/train/labels")))
print('驗證集圖片數量 : ', len(os.listdir("./datasets/val/images")))
print('驗證集標記數量 : ', len(os.listdir("./datasets/val/labels")))

# ---------------------------
# 模型訓練
# ---------------------------
model = YOLO('yolo12n.pt')  # 初次訓練用官方模型
results = model.train(
    data="./aortic_valve_colab.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
    device=0
)

# ---------------------------
# 測試集處理與推論
# ---------------------------
unzip_if_needed("testing_image.zip", "./testing_image")
TEST_ROOT = find_patient_root("./testing_image")

DST_TEST = "./datasets/test/images"
os.makedirs(DST_TEST, exist_ok=True)

# 收集所有 patient 圖片
all_files = []
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
    if os.path.exists(dst): os.remove(dst)
    shutil.copy2(f, dst)
    copied += 1

print(f"來源根目錄：{TEST_ROOT}")
print(f"完成複製！總共 {copied} 張圖片。")
print('測試集圖片數量 : ', len(os.listdir(DST_TEST)))

# ---------------------------
# 推論（Predict）
# ---------------------------
model = YOLO('./runs/detect/train/weights/best.pt')  # 修改為最新 best.pt 路徑
results = model.predict(
    source=DST_TEST,
    save=True,
    imgsz=640,
    device=0
)

print(f"共偵測 {len(results)} 張圖片。")
print('第 260 張的預測類別 : ', results[260].boxes.cls[0].item())
print('預測信心分數 : ', results[260].boxes.conf[0].item())
print('預測框座標 : ', results[260].boxes.xyxy[0].tolist())

# ---------------------------
# 儲存預測結果到文字檔
# ---------------------------
os.makedirs('./predict_txt', exist_ok=True)
with open('./predict_txt/images.txt', 'w', encoding='utf-8') as output_file:
    for i in range(len(results)):
        filename = str(results[i].path).replace('\\', '/').split('/')[-1].split('.png')[0]
        boxes = results[i].boxes
        box_num = len(boxes.cls.tolist())
        if box_num > 0:
            for j in range(box_num):
                label = int(boxes.cls[j].item())
                conf = boxes.conf[j].item()
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                output_file.write(line)

# ---------------------------
# 清理記憶體
# ---------------------------
del boxes, all_files, results
gc.collect()
torch.cuda.empty_cache()
print("✅ 程式執行完畢，GPU 記憶體已釋放。")

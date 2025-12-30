# week9.py — YOLO 資料集可讀報告版
# -------------------------------------------------------
# 會輸出：
# 1) 終端機可讀摘要
# 2) dataset_stats/dataset_report.md  (Markdown)
# 3) dataset_stats/dataset_report.html (HTML 可直接用瀏覽器開)
# -------------------------------------------------------

import os
import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from PIL import Image
from datetime import datetime

# ==== 路徑（保持你之前設定）====
TRAIN_IMAGES = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\images"
TRAIN_LABELS = r"C:\Users\Lab902\Desktop\week8\pc\datasets\train\labels"
VAL_IMAGES   = r"C:\Users\Lab902\Desktop\week8\pc\datasets\val\images"
VAL_LABELS   = r"C:\Users\Lab902\Desktop\week8\pc\datasets\val\labels"

# 類別名稱（如有多類就改成多個）
CLASS_NAMES = ["aortic_valve"]

# 熱力圖網格
GRID_N = 10

# 輸出檔案
OUT_DIR   = "./dataset_stats"
MD_PATH   = os.path.join(OUT_DIR, "dataset_report.md")
HTML_PATH = os.path.join(OUT_DIR, "dataset_report.html")


@dataclass
class ImageRecord:
    split: str
    image_path: str
    width: int
    height: int
    num_labels: int
    is_positive: int


@dataclass
class BBoxRecord:
    split: str
    image_path: str
    cls: int
    x_center: float
    y_center: float
    w: float
    h: float
    x_center_px: float
    y_center_px: float
    w_px: float
    h_px: float
    area_norm: float
    area_px: float


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    if not os.path.isdir(images_dir):
        return files
    for fn in os.listdir(images_dir):
        if os.path.splitext(fn.lower())[1] in exts:
            files.append(os.path.join(images_dir, fn))
    files.sort()
    return files


def yolo_txt_for_image(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO line: {line}")
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    return cls, x, y, w, h


def analyze_split(split_name: str, images_dir: str, labels_dir: str) -> Tuple[List[ImageRecord], List[BBoxRecord], Counter]:
    img_records: List[ImageRecord] = []
    bbox_records: List[BBoxRecord] = []
    heat_counter: Counter = Counter()

    images = list_images(images_dir)
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"[WARN] 跳過無法開啟圖片: {img_path} ({e})")
            continue

        txt = yolo_txt_for_image(img_path, labels_dir)
        n_labels = 0

        if os.path.exists(txt):
            try:
                with open(txt, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            except UnicodeDecodeError:
                with open(txt, "r", encoding="utf-8-sig") as f:
                    lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            except Exception as e:
                print(f"[WARN] 讀取標註失敗: {txt} ({e})")
                lines = []

            for ln in lines:
                try:
                    cls, x, y, w, h = parse_yolo_line(ln)
                except Exception as e:
                    print(f"[WARN] 跳過錯誤標註: {txt} -> {ln} ({e})")
                    continue
                n_labels += 1

                # 轉像素
                x_px, y_px = x * W, y * H
                w_px, h_px = w * W, h * H
                area_norm = w * h
                area_px = w_px * h_px

                bbox_records.append(BBoxRecord(
                    split=split_name, image_path=img_path, cls=cls,
                    x_center=x, y_center=y, w=w, h=h,
                    x_center_px=x_px, y_center_px=y_px, w_px=w_px, h_px=h_px,
                    area_norm=area_norm, area_px=area_px
                ))

                gx = min(GRID_N - 1, max(0, int(x * GRID_N)))
                gy = min(GRID_N - 1, max(0, int(y * GRID_N)))
                heat_counter[(gx, gy)] += 1

        img_records.append(ImageRecord(
            split=split_name, image_path=img_path, width=W, height=H,
            num_labels=n_labels, is_positive=1 if n_labels > 0 else 0
        ))
    return img_records, bbox_records, heat_counter


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return dict(count=0, mean=0, min=0, max=0, std=0)
    n = len(values)
    mean = sum(values) / n
    vmin = min(values); vmax = max(values)
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return dict(count=n, mean=mean, min=vmin, max=vmax, std=std)


def md_table(rows: List[Dict], headers: List[str]) -> str:
    if not rows:
        return "_(無資料)_\n"
    head = "| " + " | ".join(headers) + " |\n"
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"
    return head + sep + body


def html_table(rows: List[Dict], headers: List[str]) -> str:
    if not rows:
        return "<p><em>無資料</em></p>"
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{r.get(h, '')}</td>" for h in headers)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def render_heatmap_ascii(heat: Counter) -> str:
    rows = []
    max_val = max(heat.values()) if heat else 1
    for gy in range(GRID_N):
        row = []
        for gx in range(GRID_N):
            v = heat.get((gx, gy), 0)
            ratio = v / max_val if max_val > 0 else 0
            ch = " "
            if ratio > 0.75: ch = "█"
            elif ratio > 0.5: ch = "▓"
            elif ratio > 0.25: ch = "▒"
            elif ratio > 0: ch = "░"
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows)


def render_heatmap_html(heat: Counter) -> str:
    max_val = max(heat.values()) if heat else 1
    cells = []
    for gy in range(GRID_N):
        tds = []
        for gx in range(GRID_N):
            v = heat.get((gx, gy), 0)
            ratio = v / max_val if max_val > 0 else 0
            b = int(255 - 180 * ratio)  # 白→深藍
            color = f"rgb(40,70,{b})"
            tds.append(f'<td title="({gx},{gy})={v}" style="background:{color};color:#fff;text-align:center">{v}</td>')
        cells.append(f"<tr>{''.join(tds)}</tr>")
    return f"""
    <table class="heat">
      <tbody>
        {''.join(cells)}
      </tbody>
    </table>
    <small>（左上角= (0,0)；右下角= ({GRID_N-1},{GRID_N-1}) ）</small>
    """


def main():
    ensure_dir(OUT_DIR)

    splits = [
        ("train", TRAIN_IMAGES, TRAIN_LABELS),
        ("val",   VAL_IMAGES,   VAL_LABELS)
    ]

    all_img: List[ImageRecord] = []
    all_box: List[BBoxRecord] = []
    heat_total: Counter = Counter()
    split_summaries = []
    split_tables_md = []
    split_tables_html = []

    # 分析每個 split
    for split_name, img_dir, lbl_dir in splits:
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            print(f"[INFO] 跳過 {split_name}（資料夾不存在）：{img_dir} / {lbl_dir}")
            continue

        img_recs, bbox_recs, heat = analyze_split(split_name, img_dir, lbl_dir)
        all_img.extend(img_recs)
        all_box.extend(bbox_recs)
        heat_total += heat

        pos = sum(r.is_positive for r in img_recs)
        neg = len(img_recs) - pos
        labels_per_img = [r.num_labels for r in img_recs]
        mean_labels = (sum(labels_per_img) / len(labels_per_img)) if labels_per_img else 0.0

        # BBox 統計
        if bbox_recs:
            w_norm = [r.w for r in bbox_recs]; h_norm = [r.h for r in bbox_recs]
            w_px = [r.w_px for r in bbox_recs]; h_px = [r.h_px for r in bbox_recs]
            area_n = [r.area_norm for r in bbox_recs]; area_p = [r.area_px for r in bbox_recs]
            bbox_stats = {
                "w_norm": summarize_numeric(w_norm),
                "h_norm": summarize_numeric(h_norm),
                "w_px": summarize_numeric(w_px),
                "h_px": summarize_numeric(h_px),
                "area_norm": summarize_numeric(area_n),
                "area_px": summarize_numeric(area_p),
            }
        else:
            bbox_stats = {k: summarize_numeric([]) for k in ["w_norm","h_norm","w_px","h_px","area_norm","area_px"]}

        # 終端機摘要
        print(f"\n[{split_name.upper()}] 圖片 {len(img_recs)} 張｜正樣本 {pos}｜負樣本 {neg}｜每圖框數均值 {mean_labels:.3f}")
        print("  BBox 寬度(px):", bbox_stats["w_px"])
        print("  BBox 高度(px):", bbox_stats["h_px"])
        print("  BBox 面積(px):", bbox_stats["area_px"])

        # 報告表格（節選前 10 筆，避免爆量）
        split_summaries.append({
            "split": split_name,
            "images": len(img_recs),
            "positive": pos,
            "negative": neg,
            "labels_per_image_mean": f"{mean_labels:.3f}"
        })
        sample_img_rows = [asdict(r) for r in img_recs[:10]]
        sample_box_rows = [asdict(r) for r in bbox_recs[:10]]

        # Markdown
        split_tables_md.append(
            f"### Split: {split_name}\n"
            f"**摘要**\n\n"
            + md_table([split_summaries[-1]], ["split","images","positive","negative","labels_per_image_mean"])
            + "\n**部分影像清單（前 10 筆）**\n\n"
            + md_table(sample_img_rows, list(sample_img_rows[0].keys()) if sample_img_rows else ["split","image_path","width","height","num_labels","is_positive"])
            + "\n**部分 BBox（前 10 筆）**\n\n"
            + md_table(sample_box_rows, list(sample_box_rows[0].keys()) if sample_box_rows else ["split","image_path","cls","x_center","y_center","w","h","x_center_px","y_center_px","w_px","h_px","area_norm","area_px"])
        )

        # HTML
        split_tables_html.append(
            f"""
            <h3>Split: {split_name}</h3>
            <h4>摘要</h4>
            {html_table([split_summaries[-1]], ["split","images","positive","negative","labels_per_image_mean"])}
            <h4>部分影像清單（前 10 筆）</h4>
            {html_table(sample_img_rows, list(sample_img_rows[0].keys()) if sample_img_rows else ["split","image_path","width","height","num_labels","is_positive"])}
            <h4>部分 BBox（前 10 筆）</h4>
            {html_table(sample_box_rows, list(sample_box_rows[0].keys()) if sample_box_rows else ["split","image_path","cls","x_center","y_center","w","h","x_center_px","y_center_px","w_px","h_px","area_norm","area_px"])}
            """
        )

    # 整體摘要
    total_images = len(all_img)
    pos_total = sum(r.is_positive for r in all_img)
    neg_total = total_images - pos_total
    per_img_labels = [r.num_labels for r in all_img]
    mean_labels_all = (sum(per_img_labels) / len(per_img_labels)) if per_img_labels else 0.0

    if all_box:
        w_norm = [r.w for r in all_box]; h_norm = [r.h for r in all_box]
        w_px = [r.w_px for r in all_box]; h_px = [r.h_px for r in all_box]
        area_n = [r.area_norm for r in all_box]; area_p = [r.area_px for r in all_box]
        overall_bbox_stats = {
            "w_norm": summarize_numeric(w_norm),
            "h_norm": summarize_numeric(h_norm),
            "w_px": summarize_numeric(w_px),
            "h_px": summarize_numeric(h_px),
            "area_norm": summarize_numeric(area_n),
            "area_px": summarize_numeric(area_p),
        }
    else:
        overall_bbox_stats = {k: summarize_numeric([]) for k in ["w_norm","h_norm","w_px","h_px","area_norm","area_px"]}

    # 供 HTML 表格使用的 BBox 統計列（修正 set/dict bug）
    bbox_rows_html = [
        {
            "metric": "w_px",
            "count": overall_bbox_stats["w_px"]["count"],
            "mean": f"{overall_bbox_stats['w_px']['mean']:.2f}",
            "min":  f"{overall_bbox_stats['w_px']['min']:.2f}",
            "max":  f"{overall_bbox_stats['w_px']['max']:.2f}",
            "std":  f"{overall_bbox_stats['w_px']['std']:.2f}",
        },
        {
            "metric": "h_px",
            "count": overall_bbox_stats["h_px"]["count"],
            "mean": f"{overall_bbox_stats['h_px']['mean']:.2f}",
            "min":  f"{overall_bbox_stats['h_px']['min']:.2f}",
            "max":  f"{overall_bbox_stats['h_px']['max']:.2f}",
            "std":  f"{overall_bbox_stats['h_px']['std']:.2f}",
        },
        {
            "metric": "area_px",
            "count": overall_bbox_stats["area_px"]["count"],
            "mean": f"{overall_bbox_stats['area_px']['mean']:.2f}",
            "min":  f"{overall_bbox_stats['area_px']['min']:.2f}",
            "max":  f"{overall_bbox_stats['area_px']['max']:.2f}",
            "std":  f"{overall_bbox_stats['area_px']['std']:.2f}",
        },
    ]

    # ===== Markdown 報告 =====
    ensure_dir(OUT_DIR)
    md = []
    md.append(f"# YOLO 資料集報告\n\n產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("## 整體摘要\n")
    md.append(md_table([{
        "images": total_images,
        "positive": pos_total,
        "negative": neg_total,
        "labels_per_image_mean": f"{mean_labels_all:.3f}",
        "classes": ", ".join(CLASS_NAMES)
    }], ["images","positive","negative","labels_per_image_mean","classes"]))

    md.append("\n**BBox 統計（整體）**\n\n")
    md.append(md_table([
        {"metric":"w_px", "count":overall_bbox_stats["w_px"]["count"], "mean":f"{overall_bbox_stats['w_px']['mean']:.2f}", "min":f"{overall_bbox_stats['w_px']['min']:.2f}", "max":f"{overall_bbox_stats['w_px']['max']:.2f}", "std":f"{overall_bbox_stats['w_px']['std']:.2f}"},
        {"metric":"h_px", "count":overall_bbox_stats["h_px"]["count"], "mean":f"{overall_bbox_stats['h_px']['mean']:.2f}", "min":f"{overall_bbox_stats['h_px']['min']:.2f}", "max":f"{overall_bbox_stats['h_px']['max']:.2f}", "std":f"{overall_bbox_stats['h_px']['std']:.2f}"},
        {"metric":"area_px", "count":overall_bbox_stats["area_px"]["count"], "mean":f"{overall_bbox_stats['area_px']['mean']:.2f}", "min":f"{overall_bbox_stats['area_px']['min']:.2f}", "max":f"{overall_bbox_stats['area_px']['max']:.2f}", "std":f"{overall_bbox_stats['area_px']['std']:.2f}"},
    ], ["metric","count","mean","min","max","std"]))

    md.append("\n## 目標中心熱力圖（10×10，ASCII）\n")
    md.append("```\n" + render_heatmap_ascii(heat_total) + "\n```\n")
    md.append("\n---\n")
    md.append("\n".join(split_tables_md))

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ===== HTML 報告 =====
    html = f"""<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8">
<title>YOLO 資料集報告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans CJK TC", "Microsoft JhengHei", Arial, sans-serif; margin: 24px; line-height: 1.6; }}
h1,h2,h3 {{ margin-top: 1.2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #e5e5e5; padding: 6px 8px; font-size: 14px; }}
thead th {{ background: #f7f7f7; }}
small {{ color: #999; }}
.heat td {{ width: 26px; height: 26px; padding: 0; font-size: 12px; }}
.section {{ margin-bottom: 28px; }}
.kv {{ display: grid; grid-template-columns: 160px 1fr; gap: 8px 12px; max-width: 560px; }}
.kv div:nth-child(odd) {{ color: #666; }}
</style>
</head><body>
<h1>YOLO 資料集報告</h1>
<p><small>產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>

<div class="section">
  <h2>整體摘要</h2>
  {html_table([{
      "images": total_images,
      "positive": pos_total,
      "negative": neg_total,
      "labels_per_image_mean": f"{mean_labels_all:.3f}",
      "classes": ", ".join(CLASS_NAMES)
  }], ["images","positive","negative","labels_per_image_mean","classes"])}
</div>

<div class="section">
  <h2>BBox 統計（整體）</h2>
  {html_table(bbox_rows_html, ["metric","count","mean","min","max","std"])}
</div>

<div class="section">
  <h2>目標中心熱力圖（10×10）</h2>
  {render_heatmap_html(heat_total)}
</div>

<div class="section">
  <h2>各 Split 詳細（節選）</h2>
  {''.join(split_tables_html)}
</div>

</body></html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # ===== 終端機最終提示 =====
    print("\n== 完成！人可讀報告已產生 ==")
    print("Markdown：", os.path.abspath(MD_PATH))
    print("HTML    ：", os.path.abspath(HTML_PATH))
    print("\n直接用瀏覽器打開 HTML，或把 Markdown 複製到簡報即可。")


if __name__ == "__main__":
    main()

import fitz  # PyMuPDF
import cv2
import numpy as np
import json
from pathlib import Path

from drawing_contrast import paths


def pdf_page_to_image(doc, page_num, zoom=2):
    """将PDF页面转换为OpenCV图像"""
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def merge_overlapping_regions(regions, merge_distance=50):
    """
    合并重叠或相邻的矩形区域
    """
    if not regions:
        return []

    # 转换为numpy数组便于计算
    rects = np.array(regions, dtype=np.int32)

    # 扩展每个矩形以捕获相邻区域
    expanded = []
    for x, y, w, h in rects:
        expanded.append([
            max(0, x - merge_distance),
            max(0, y - merge_distance),
            w + 2 * merge_distance,
            h + 2 * merge_distance
        ])
    expanded = np.array(expanded, dtype=np.int32)

    # 转换为(x1, y1, x2, y2)格式用于groupRectangles
    boxes = []
    for x, y, w, h in expanded:
        boxes.append([x, y, x + w, y + h])
    boxes = np.array(boxes, dtype=np.int32)

    # 使用OpenCV的groupRectangles合并重叠框
    grouped, weights = cv2.groupRectangles(list(boxes), groupThreshold=0, eps=0.1)

    if len(grouped) == 0:
        return regions

    # 转换回(x, y, w, h)格式，并缩回原始大小
    merged = []
    for x1, y1, x2, y2 in grouped:
        x = max(0, x1 + merge_distance)
        y = max(0, y1 + merge_distance)
        w = max(10, x2 - x1 - 2 * merge_distance)
        h = max(10, y2 - y1 - 2 * merge_distance)
        merged.append((x, y, w, h))

    return merged


def find_difference_regions(img1, img2, min_area=500, max_area_ratio=0.3, blur_size=5, threshold=30, top_n=None):
    """
    对比两张图像，返回差异区域的边界框列表
    img1: 原始图纸图像
    img2: 衍生图纸图像
    """
    # 确保两张图像尺寸一致
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = min(h1, h2)
    target_w = min(w1, w2)
    total_area = target_w * target_h

    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))

    # 转为灰度图
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，减少微小噪声差异
    gray1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 0)
    gray2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 0)

    # 计算绝对差异
    diff = cv2.absdiff(gray1, gray2)

    # 二值化
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 形态学操作，连接相邻的差异区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > total_area * max_area_ratio:
            # 过滤掉过大的区域（比如整张图）
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 添加一些边距
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(target_w - x, w + 2 * padding)
        h = min(target_h - y, h + 2 * padding)
        regions.append((x, y, w, h))

    # 合并相邻/重叠的区域
    regions = merge_overlapping_regions(regions, merge_distance=50)

    # 按面积从大到小排序
    regions_with_area = []
    for r in regions:
        x, y, w, h = r
        regions_with_area.append((w * h, r))
    regions_with_area.sort(key=lambda x: x[0], reverse=True)

    # 只保留前N个最大的差异区域（如果指定了）
    if top_n is not None:
        regions_with_area = regions_with_area[:top_n]

    regions = [r for _, r in regions_with_area]

    return regions, (target_w, target_h)


def crop_image_to_focus(img, crop_margins):
    """
    裁剪图像，只保留主体区域
    crop_margins: dict with keys 'top', 'bottom', 'left', 'right' (ratios 0.0~1.0)
    Returns: cropped_img, (offset_x, offset_y)
    """
    h, w = img.shape[:2]
    top = int(h * crop_margins.get('top', 0))
    bottom = int(h * (1 - crop_margins.get('bottom', 0)))
    left = int(w * crop_margins.get('left', 0))
    right = int(w * (1 - crop_margins.get('right', 0)))
    cropped = img[top:bottom, left:right]
    return cropped, (left, top)


def annotate_pdf(input_pdf_path, output_pdf_path, original_pdf_path, zoom=2, min_area=500,
                 focus_area=None):
    """
    在衍生图纸上标注与原始图纸的差异
    focus_area: 可选，裁剪参数 dict，例如 {'top':0.08, 'bottom':0.18, 'left':0.08, 'right':0.05}
                表示只对比页面中间的主体区域，过滤掉标题栏、边框等
    """
    doc_derived = fitz.open(input_pdf_path)
    doc_original = fitz.open(original_pdf_path)

    max_pages = max(len(doc_derived), len(doc_original))

    for page_idx in range(max_pages):
        if page_idx >= len(doc_derived):
            break

        page = doc_derived.load_page(page_idx)

        if page_idx >= len(doc_original):
            # 原始图纸没有这一页，整页标注为新增
            rect = page.rect
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=(1, 0, 0))  # 红色
            annot.set_border(width=3)
            annot.update()
            continue

        # 获取两张图对应页面的图像
        img_derived_full = pdf_page_to_image(doc_derived, page_idx, zoom)
        img_original_full = pdf_page_to_image(doc_original, page_idx, zoom)

        # 如果指定了 focus_area，裁剪主体区域
        if focus_area:
            img_derived, (offset_x, offset_y) = crop_image_to_focus(img_derived_full, focus_area)
            img_original, _ = crop_image_to_focus(img_original_full, focus_area)
        else:
            img_derived = img_derived_full
            img_original = img_original_full
            offset_x, offset_y = 0, 0

        # 找出差异区域
        regions, (img_w, img_h) = find_difference_regions(img_original, img_derived, min_area=min_area, top_n=30)

        if not regions:
            print(f"第 {page_idx + 1} 页未检测到显著差异")
            continue

        print(f"第 {page_idx + 1} 页检测到 {len(regions)} 处差异")

        # 将图像坐标转换为PDF页面坐标（加上裁剪偏移）
        page_rect = page.rect
        full_h, full_w = img_derived_full.shape[:2]
        scale_x = page_rect.width / full_w
        scale_y = page_rect.height / full_h

        for x, y, w, h in regions:
            # 加上裁剪偏移量
            abs_x = x + offset_x
            abs_y = y + offset_y
            pdf_x = abs_x * scale_x
            pdf_y = abs_y * scale_y
            pdf_w = w * scale_x
            pdf_h = h * scale_y

            rect = fitz.Rect(pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h)
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=(1, 0, 0))  # 红色
            annot.set_border(width=2)
            annot.update()

    doc_derived.save(output_pdf_path)
    doc_derived.close()
    doc_original.close()
    print(f"标注完成，已保存至: {output_pdf_path}")


if __name__ == "__main__":
    base_dir = paths.program_dir()
    original_pdf = base_dir / "8DC-2-4382-640-01.pdf"
    derived_pdf = base_dir / "8DC-2-4383-483-01.pdf"
    output_pdf = base_dir / "8DC-2-4383-483-01_标注差异.pdf"

    # 对比模式1：全页对比（默认）
    # annotate_pdf(
    #     input_pdf_path=str(derived_pdf),
    #     output_pdf_path=str(output_pdf),
    #     original_pdf_path=str(original_pdf),
    #     zoom=2,
    #     min_area=500
    # )

    # 对比模式2：只对比图纸主体区域（AI自动识别）
    # 先运行 detect_focus.py 生成 focus_area.json
    focus_json = base_dir / "focus_area.json"
    if focus_json.exists():
        with open(focus_json, "r", encoding="utf-8") as f:
            ai_focus = json.load(f)
        # AI返回格式: {"left": 0.15, "top": 0.08, "right": 0.85, "bottom": 0.75}
        # 转换为 crop_margins 格式
        focus_area = {
            'left': ai_focus.get('left', 0),
            'top': ai_focus.get('top', 0),
            'right': 1.0 - ai_focus.get('right', 1.0),
            'bottom': 1.0 - ai_focus.get('bottom', 1.0)
        }
        print(f"使用AI识别的主体区域: {focus_area}")
    else:
        # 兜底：手动比例
        focus_area = {'top': 0.08, 'bottom': 0.18, 'left': 0.08, 'right': 0.05}
        print(f"未找到 focus_area.json，使用手动比例: {focus_area}")

    annotate_pdf(
        input_pdf_path=str(derived_pdf),
        output_pdf_path=str(output_pdf),
        original_pdf_path=str(original_pdf),
        zoom=2,
        min_area=500,
        focus_area=focus_area
    )

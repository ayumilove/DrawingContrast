"""
生成图纸差异图层：
  - img_a.png：图纸A（衍生后的新图纸）
  - img_b.png：图纸B（原始图纸）
  - diff_added.png：A比B新增的元素（红色，带透明通道）
  - diff_removed.png：A比B删除的元素（蓝色，带透明通道）
  - mask.png：主体区域掩码（白色=主体，黑色=边框区域）

用法：uv run generate_diff_layers.py
"""

import fitz
import cv2
import numpy as np
import json
import os
import base64
import requests
import threading
from pathlib import Path

from drawing_contrast import paths
from drawing_contrast.core.view_blocks import block_level_align
from drawing_contrast.core.vlm_analyzer import extract_diff_regions, analyze_regions, summarize_changes
from drawing_contrast.ai_config import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completions_url


def pdf_page_to_image(pdf_path, page_num=0, zoom=2, thicken_lines=True):
    """将PDF页面转为OpenCV BGR图像
    thicken_lines: 是否加粗线条（模拟Chrome渲染效果）
    """
    doc = fitz.open(pdf_path)
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
    doc.close()

    if thicken_lines:
        # 轻微加粗线条：对非白色像素（线条）做 1次 dilate
        # 使用椭圆核(2x2)，效果约等于线条加粗0.5px，接近Chrome渲染效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 提取线条区域（深色像素）
        line_mask = (gray < 200).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_mask_thick = cv2.dilate(line_mask, kernel, iterations=1)
        # 将加粗后的线条写回图像（保持原有颜色）
        # 新增的像素用邻近线条颜色填充
        new_pixels = cv2.bitwise_and(line_mask_thick, cv2.bitwise_not(line_mask))
        # 对原图做同样的dilate来获取颜色
        img_dilated = cv2.dilate(img, kernel, iterations=1)
        # 只在新增像素位置用dilated版本替换
        mask_3ch = cv2.merge([new_pixels, new_pixels, new_pixels])
        img = np.where(mask_3ch > 0, img_dilated, img)

    return img


def pdf_page_to_base64(pdf_path, page_num=0, zoom=2):
    """将PDF页面转为base64 PNG"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def detect_focus_area_ai(image_base64, api_key, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL):
    """调用多模态大模型识别图纸主体区域，含自动重试"""
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            coords = _call_ai_once(image_base64, api_key, model, base_url)
            # 校验字段合理性
            for key in ('left', 'top', 'right', 'bottom'):
                v = coords[key]
                if not (0 <= v <= 1):
                    raise ValueError(f"字段 {key}={v} 超出 [0,1] 范围")
            if coords['left'] >= coords['right'] or coords['top'] >= coords['bottom']:
                raise ValueError(f"坐标逻辑错误: left >= right 或 top >= bottom")
            return coords
        except Exception as e:
            last_error = e
            print(f"AI识别第{attempt+1}次失败: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
    # 全部失败，返回安全的默认区域
    print(f"AI识别{max_retries}次均失败，使用默认主体区域")
    return {"left": 0.05, "top": 0.05, "right": 0.95, "bottom": 0.80}


def _call_ai_once(image_base64, api_key, model, base_url=DEFAULT_BASE_URL):
    """单次 AI 调用"""
    url = chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = (
        "你是一位机械制图专家。请仔细分析这张工程图纸，"
        "识别出图纸中【零件主体视图】所在的精确区域。"
        "主体视图包括：零件的正视图、侧视图、剖视图、局部放大图、尺寸标注线等。"
        "需要排除的内容：页面外边框、顶部标题栏/表头、左侧信息栏（设计/审核/日期等）、"
        "底部公司名称/零件号栏、右侧分区标记（A/B/C/D/E）、纯文字注释块。"
        "请只返回一个JSON对象，格式严格如下，不要有任何其他文字说明：\n"
        '{"left": 0.12, "top": 0.08, "right": 0.92, "bottom": 0.75}'
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}],
        "max_tokens": 256,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    print(f"AI模型返回: {content[:300]}")

    return _parse_focus_json(content)


def _parse_focus_json(content):
    """从 AI 返回文本中提取主体区域 JSON"""
    import re
    # 去掉可能残留的 <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # 提取 ```json ... ``` 代码块
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        text = code_match.group(1).strip()
    # 方法1: 精确匹配含 left/bottom 的 JSON
    json_match = re.search(r'\{[\s\S]*?"left"[\s\S]*?"bottom"[\s\S]*?\}', text)
    if json_match:
        return json.loads(json_match.group())
    # 方法2: 匹配任何 JSON 对象
    json_match = re.search(r'\{[^{}]*\}', text)
    if json_match:
        obj = json.loads(json_match.group())
        if 'left' in obj and 'bottom' in obj:
            return obj
    # 方法3: 直接尝试从4个浮点数构建
    nums = re.findall(r'0\.\d+', text)
    if len(nums) >= 4:
        return {
            "left": float(nums[0]), "top": float(nums[1]),
            "right": float(nums[2]), "bottom": float(nums[3])
        }
    raise ValueError(f"无法解析AI返回: {content[:200]}")


def create_focus_mask(img_shape, focus_area):
    """
    创建主体区域掩码
    focus_area: {"left": 0.15, "top": 0.08, "right": 0.85, "bottom": 0.75}
    返回: 单通道掩码，主体区域=255，边框区域=0
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = int(w * focus_area["left"])
    y1 = int(h * focus_area["top"])
    x2 = int(w * focus_area["right"])
    y2 = int(h * focus_area["bottom"])
    mask[y1:y2, x1:x2] = 255
    return mask


def align_images(img_a, img_b, method='ecc'):
    """
    图像配准：将 img_b 对齐到 img_a 的坐标系
    解决两张图纸之间的平移/旋转偏移问题
    
    method:
      'ecc'      - ECC增强相关系数法（亚像素精度，适合小偏移）
      'feature'  - 特征点匹配法（适合较大偏移）
      'phase'    - 相位相关法（仅平移，速度最快）
    
    返回: (对齐后的 img_b, different_drawings: bool)
           different_drawings=True 表示检测到两张图是完全不同的图纸（非版本变更）
    """
    h, w = img_a.shape[:2]
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    if method == 'phase':
        # 相位相关：只检测平移
        shift, _ = cv2.phaseCorrelate(
            gray_a.astype(np.float64),
            gray_b.astype(np.float64)
        )
        dx, dy = shift
        print(f"  相位相关检测偏移: dx={dx:.2f}, dy={dy:.2f} 像素")
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(img_b, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return aligned, False

    elif method == 'ecc':
        # ECC法：亚像素精度的仿射对齐
        # 先用相位相关获得初始偏移估计
        shift, _ = cv2.phaseCorrelate(
            gray_a.astype(np.float64),
            gray_b.astype(np.float64)
        )
        dx, dy = shift
        print(f"  初始偏移估计: dx={dx:.2f}, dy={dy:.2f} 像素")

        # 用仿射模型（平移+旋转+缩放）
        warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

        try:
            # 在缩小版本上运行ECC（性能优化）
            scale = 0.25
            small_a = cv2.resize(gray_a, None, fx=scale, fy=scale)
            small_b = cv2.resize(gray_b, None, fx=scale, fy=scale)
            small_matrix = np.float32([[1, 0, dx * scale], [0, 1, dy * scale]])

            _, small_matrix = cv2.findTransformECC(
                small_a, small_b, small_matrix,
                cv2.MOTION_AFFINE, criteria,
                inputMask=None, gaussFiltSize=5
            )

            # 缩放回原始尺寸
            warp_matrix = small_matrix.copy()
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale

            print(f"  ECC精细对齐完成:")
            print(f"    平移: dx={warp_matrix[0,2]:.2f}, dy={warp_matrix[1,2]:.2f}")
            print(f"    旋转/缩放矩阵: [[{warp_matrix[0,0]:.6f}, {warp_matrix[0,1]:.6f}],")
            print(f"                     [{warp_matrix[1,0]:.6f}, {warp_matrix[1,1]:.6f}]]")
        except cv2.error as e:
            print(f"  ECC对齐失败，使用相位相关结果: {e}")
            warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        aligned = cv2.warpAffine(img_b, warp_matrix, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return aligned, False

    elif method == 'feature':
        # 特征点匹配法
        detector = cv2.AKAZE_create()
        kp_a, desc_a = detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = detector.detectAndCompute(gray_b, None)

        if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            print("  特征点不足，跳过对齐")
            return img_b, False

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desc_b, desc_a, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        print(f"  特征匹配: {len(kp_a)} / {len(kp_b)} 关键点, {len(good)} 有效匹配")

        if len(good) < 10:
            print("  匹配点不足，跳过对齐")
            return img_b, True

        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 估算仿射变换（RANSAC鲁棒估计）
        M, inliers = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M is None:
            print("  无法估算变换矩阵，跳过对齐")
            return img_b, True

        inlier_count = np.sum(inliers)
        inlier_ratio = inlier_count / len(good) if len(good) > 0 else 0
        translation = np.sqrt(M[0, 2] ** 2 + M[1, 2] ** 2)
        scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        max_dim = max(w, h)
        print(f"  RANSAC内点: {inlier_count}/{len(good)} (比例: {inlier_ratio:.2%})")
        print(f"  变换矩阵: 平移=({M[0,2]:.2f}, {M[1,2]:.2f}), "
              f"旋转={np.degrees(np.arctan2(M[1,0], M[0,0])):.4f}°, "
              f"缩放={scale:.6f}")

        # ── 可信度验证：防止错误匹配导致图像被变换到可视区域外 ──
        # 条件1: RANSAC 内点比例太低（大部分匹配都是错误的）
        if inlier_count < 20 or inlier_ratio < 0.1:
            print(f"  ⚠️ RANSAC内点不足（{inlier_count}个, 比例{inlier_ratio:.1%}），判定为不同图纸，跳过对齐")
            return img_b, True
        # 条件2: 平移量过大（超过图像尺寸的30%，说明图纸完全不同）
        if translation > max_dim * 0.3:
            print(f"  ⚠️ 平移量过大（{translation:.1f}px > {max_dim * 0.3:.1f}px），判定为不同图纸，跳过对齐")
            return img_b, True
        # 条件3: 缩放比例偏离1.0太多（同一零件不同版本的缩放应接近1.0）
        if scale < 0.7 or scale > 1.3:
            print(f"  ⚠️ 缩放比例异常（{scale:.4f}），判定为不同图纸，跳过对齐")
            return img_b, True

        aligned = cv2.warpAffine(img_b, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return aligned, False

    return img_b, False


def generate_diff_layers(img_a, img_b, focus_mask, blur_size=7, threshold=25):
    """
    生成差异图层
    img_a: 新图纸（A，衍生后的）
    img_b: 原始图纸（B）
    focus_mask: 主体区域掩码
    返回: (diff_added_rgba, diff_removed_rgba)
      - diff_added: A有B无的像素（红色RGBA）
      - diff_removed: B有A无的像素（蓝色RGBA）
    """
    h, w = img_a.shape[:2]

    # 统一尺寸
    img_b_resized = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 转灰度
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减噪（核变大，更好地消除渲染差异）
    gray_a_blur = cv2.GaussianBlur(gray_a, (blur_size, blur_size), 0)
    gray_b_blur = cv2.GaussianBlur(gray_b, (blur_size, blur_size), 0)

    # 自适应二值化（白底黑线图纸：像素值低=有内容，像素值高=空白）
    # 使用更严格的阈值，只保留真正有墨的线条
    _, bin_a = cv2.threshold(gray_a_blur, 180, 255, cv2.THRESH_BINARY_INV)
    _, bin_b = cv2.threshold(gray_b_blur, 180, 255, cv2.THRESH_BINARY_INV)

    # A有B无 = 新增（红色）
    added = cv2.bitwise_and(bin_a, cv2.bitwise_not(bin_b))
    # B有A无 = 删除（蓝色）
    removed = cv2.bitwise_and(bin_b, cv2.bitwise_not(bin_a))

    # 形态学处理：去噪+连接
    # 先 open 去掉孤立噪点，再 close 连接相邻像素，最后 dilate 稍微加粗差异线条使其可见
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    added = cv2.morphologyEx(added, cv2.MORPH_OPEN, kernel_open, iterations=1)
    added = cv2.morphologyEx(added, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    added = cv2.dilate(added, kernel_dilate, iterations=1)
    removed = cv2.morphologyEx(removed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    removed = cv2.morphologyEx(removed, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    removed = cv2.dilate(removed, kernel_dilate, iterations=1)

    # 应用主体区域掩码（只保留主体区域内的差异）
    added = cv2.bitwise_and(added, focus_mask)
    removed = cv2.bitwise_and(removed, focus_mask)

    # 生成红色BGRA图层（新增）—— OpenCV使用BGR通道顺序
    diff_added = np.zeros((h, w, 4), dtype=np.uint8)
    diff_added[:, :, 0] = 40   # B
    diff_added[:, :, 1] = 40   # G
    diff_added[:, :, 2] = 220  # R
    diff_added[:, :, 3] = added  # Alpha = 差异强度

    # 生成蓝色BGRA图层（删除）—— OpenCV使用BGR通道顺序
    diff_removed = np.zeros((h, w, 4), dtype=np.uint8)
    diff_removed[:, :, 0] = 220  # B
    diff_removed[:, :, 1] = 80   # G
    diff_removed[:, :, 2] = 50   # R
    diff_removed[:, :, 3] = removed  # Alpha

    return diff_added, diff_removed


def make_transparent(bgr_img, white_thresh=240):
    """将白色/近白色背景变为透明"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray >= white_thresh, 0, 255).astype(np.uint8)
    bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def _write_meta(meta, output_dir):
    """原子写 meta.json（先写临时文件再替换），避免前端轮询时读到半截文件"""
    output_dir = Path(output_dir)
    tmp = output_dir / "meta.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_dir / "meta.json")


def run_comparison(pdf_a_path, pdf_b_path, output_dir, api_key=None, use_ai=True, model=DEFAULT_MODEL, progress_callback=None, vlm_async=True, base_url=DEFAULT_BASE_URL):
    """
    执行图纸对比的核心流程
    
    参数:
        pdf_a_path: 图纸A的路径（变更后）
        pdf_b_path: 图纸B的路径（变更前）
        output_dir: 输出目录（存放图层PNG和meta.json）
        api_key: 多模态模型 API Key（用于AI识别主体区域与差异解读）
        use_ai: 是否使用AI识别主体区域
        progress_callback: 进度回调函数 callback(percent: int, message: str)
        vlm_async: True 时 VLM 区域解读转后台线程增量写 meta.json，CV 结果一出立即返回
        base_url: OpenAI 兼容 API 的 Base URL
    
    返回:
        dict: 包含对比结果元数据（vlm_async 时 regions/summary 由后台线程后续补充）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = 4

    def report(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        print(msg)

    # 1. 转换PDF为图像
    report(5, "正在转换PDF为图像...")
    img_a = pdf_page_to_image(str(pdf_a_path), zoom=zoom)
    report(20, "图纸A转换完成")
    img_b = pdf_page_to_image(str(pdf_b_path), zoom=zoom)
    report(35, "图纸B转换完成")

    # 统一尺寸（以A为基准）
    h, w = img_a.shape[:2]
    img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 2. 图像配准（全局粗配准）
    report(40, "正在对齐图纸（消除位移偏差）...")
    img_b, different_drawings = align_images(img_a, img_b, method='feature')
    if different_drawings:
        report(48, "⚠️ 检测到两张图纸为不同零件（非版本变更），已跳过对齐")
    else:
        report(48, "对齐完成")

    # 3. 识别主体区域（提前到分块之前，供分块排除边框/标题栏）
    focus_area = None
    if use_ai and api_key:
        report(50, "调用AI识别主体区域...")
        img_b64 = pdf_page_to_base64(str(pdf_a_path), zoom=2)
        for attempt in range(3):
            try:
                focus_area = detect_focus_area_ai(img_b64, api_key, model=model, base_url=base_url)
                break
            except (ValueError, Exception) as e:
                report(50, f"第{attempt+1}次尝试失败: {e}")
                if attempt == 2:
                    focus_area = None

    if focus_area is None:
        report(55, "使用默认主体区域")
        focus_area = {"left": 0.12, "top": 0.06, "right": 0.88, "bottom": 0.78}
    else:
        report(55, f"主体区域: {focus_area}")

    # 4. 生成主体区域掩码
    focus_mask = create_focus_mask(img_a.shape, focus_area)

    # 5. 视图分块逐块配准（解决三视图移位导致的满屏误报）
    block_info = {"moved_views": [], "view_stats": {"matched": 0, "added": 0, "removed": 0, "suspect": 0},
                  "low_confidence": False}
    if not different_drawings:
        report(58, "正在进行视图分块配准...")
        try:
            img_b, block_info = block_level_align(
                img_a, img_b, focus_mask=focus_mask,
                api_key=api_key if use_ai else None, model=model,
                report=lambda msg: report(60, msg),
                base_url=base_url
            )
        except Exception as e:
            report(60, f"视图分块配准失败，回退到全局配准: {e}")

    # 6. 保存透明图片（PNG 格式保证兼容性，同时生成 WebP 作为优化选项）
    img_a_transparent = make_transparent(img_a)
    img_b_transparent = make_transparent(img_b)
    # 保存 PNG 格式（所有浏览器和 QtWebEngine 都支持）
    cv2.imwrite(str(output_dir / "img_a.png"), img_a_transparent)
    cv2.imwrite(str(output_dir / "img_b.png"), img_b_transparent)
    # 同时保存 WebP 格式（体积更小，作为备选）
    webp_params = [cv2.IMWRITE_WEBP_QUALITY, 88]
    cv2.imwrite(str(output_dir / "img_a.webp"), img_a_transparent, webp_params)
    cv2.imwrite(str(output_dir / "img_b.webp"), img_b_transparent, webp_params)
    report(68, f"图像尺寸: {w}x{h}")

    # 7. 生成差异图层（PNG + WebP 双格式）
    report(70, "正在计算差异...")
    diff_added, diff_removed = generate_diff_layers(img_a, img_b, focus_mask)
    # 保存 PNG 格式（保证兼容性）
    cv2.imwrite(str(output_dir / "diff_added.png"), diff_added)
    cv2.imwrite(str(output_dir / "diff_removed.png"), diff_removed)
    # 同时保存 WebP 格式
    cv2.imwrite(str(output_dir / "diff_added.webp"), diff_added, [cv2.IMWRITE_WEBP_QUALITY, 88])
    cv2.imwrite(str(output_dir / "diff_removed.webp"), diff_removed, [cv2.IMWRITE_WEBP_QUALITY, 88])

    # 8. 差异区域聚类
    report(80, "正在提取差异区域...")
    regions = extract_diff_regions(diff_added[:, :, 3], diff_removed[:, :, 3], max_regions=20)

    # 9. 统计差异
    added_pixels = np.count_nonzero(diff_added[:, :, 3])
    removed_pixels = np.count_nonzero(diff_removed[:, :, 3])
    report(90, f"新增像素: {added_pixels}, 删除像素: {removed_pixels}")

    # 10. 先落盘纯CV结果（结果页秒开），VLM解读后台异步增量补充
    run_vlm = bool(regions and use_ai and api_key and not different_drawings)
    meta = {
        "width": w,
        "height": h,
        "focus_area": focus_area,
        "added_pixels": int(added_pixels),
        "removed_pixels": int(removed_pixels),
        "pdf_a": Path(pdf_a_path).name,
        "pdf_b": Path(pdf_b_path).name,
        "different_drawings": different_drawings,
        "regions": regions,
        "moved_views": block_info["moved_views"],
        "view_stats": block_info["view_stats"],
        "low_confidence": block_info["low_confidence"],
        "summary": "",
        "vlm_analysis": False,
        "vlm_status": "running" if run_vlm else "skipped",
        "analysis_version": 0
    }
    _write_meta(meta, output_dir)

    def vlm_task():
        """VLM 逐区域解读：每完成一个区域就增量写盘，前端轮询逐条刷新"""
        try:
            def on_region_done(done, total):
                meta["analysis_version"] += 1
                _write_meta(meta, output_dir)
                print(f"VLM解读差异区域 ({done}/{total})")

            analyze_regions(regions, img_a, img_b, api_key, model,
                            progress_callback=on_region_done, base_url=base_url)
            meta["summary"] = summarize_changes(regions, block_info["moved_views"], api_key, model,
                                                base_url=base_url)
            meta["vlm_analysis"] = True
            meta["vlm_status"] = "done"
        except Exception as e:
            meta["vlm_status"] = "failed"
            print(f"VLM区域解读失败，保留纯CV结果: {e}")
        meta["analysis_version"] += 1
        _write_meta(meta, output_dir)

    if run_vlm:
        if vlm_async:
            report(95, f"对比完成，VLM正在后台解读{len(regions)}处差异区域...")
            threading.Thread(target=vlm_task, daemon=True).start()
        else:
            report(95, f"VLM正在解读{len(regions)}处差异区域...")
            vlm_task()

    report(100, "对比完成")
    return meta


def main():
    base_dir = paths.program_dir()
    output_dir = base_dir / "web" / "static"

    pdf_a = base_dir / "8DC-2-3356-085-00.pdf"
    pdf_b = base_dir / "8DC-2-3355-314-00.pdf"
    # 从环境变量读取API Key（避免硬编码提交到仓库）
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")

    run_comparison(pdf_a, pdf_b, output_dir, api_key=api_key, use_ai=True)


if __name__ == "__main__":
    main()

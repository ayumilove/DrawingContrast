"""
VLM 差异区域解读模块：
  extract_diff_regions - 从红蓝差异掩码中聚类出差异区域列表（top N）
  analyze_regions      - 对每个区域裁高清局部图发给 VLM，返回工程语义解读
  summarize_changes    - 汇总各区域描述生成 2~3 句变更摘要

设计原则：CV 负责"框得准"，VLM 负责"说得清"。
VLM 调用失败不阻断主流程，仅降级为无描述的区域清单。
"""

import cv2
import numpy as np
import base64
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from drawing_contrast.ai_config import DEFAULT_BASE_URL, chat_completions_url

CHANGE_TYPES = ("尺寸变更", "几何变更", "标注变更", "视图移动", "噪声")


def extract_diff_regions(added_mask, removed_mask, max_regions=20, merge_dist=60, min_pixels=50):
    """
    从差异掩码聚类出差异区域
    added_mask/removed_mask: 单通道掩码（255=差异像素）
    返回: [{"bbox": [x,y,w,h], "added_pixels": n, "removed_pixels": n}, ...] 按差异像素数降序
    """
    combined = cv2.bitwise_or(added_mask, removed_mask)
    if cv2.countNonZero(combined) == 0:
        return []

    # 膨胀连接相邻差异，聚成区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_dist, merge_dist))
    clustered = cv2.dilate(combined, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(clustered, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        region_mask = (labels == i).astype(np.uint8) * 255
        added_px = cv2.countNonZero(cv2.bitwise_and(added_mask, region_mask))
        removed_px = cv2.countNonZero(cv2.bitwise_and(removed_mask, region_mask))
        if added_px + removed_px < min_pixels:
            continue
        # 用原始（未膨胀）差异像素求紧致包围盒
        orig_in_region = cv2.bitwise_and(combined, region_mask)
        ys, xs = np.nonzero(orig_in_region)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        regions.append({
            "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            "added_pixels": int(added_px),
            "removed_pixels": int(removed_px),
        })

    regions.sort(key=lambda r: r["added_pixels"] + r["removed_pixels"], reverse=True)
    regions = regions[:max_regions]
    for idx, r in enumerate(regions):
        r["id"] = idx + 1
    return regions


def _crop_region(img, bbox, pad=30):
    """按 bbox（外扩 pad）裁剪局部图"""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y1:y2, x1:x2]


def _make_labeled_pair(crop_b, crop_a, target_long=768):
    """左右拼图：左=旧图(B)，右=新图(A)，顶部标注文字，中间分隔线"""
    def resize_to(img, long_side):
        h, w = img.shape[:2]
        s = long_side / max(h, w)
        return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))

    b = resize_to(crop_b, target_long)
    a = resize_to(crop_a, target_long)
    label_h = 36
    h = max(a.shape[0], b.shape[0]) + label_h
    sep = 8
    canvas = np.full((h, b.shape[1] + sep + a.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_h:label_h + b.shape[0], :b.shape[1]] = b
    canvas[label_h:label_h + a.shape[0], b.shape[1] + sep:] = a
    canvas[:, b.shape[1]:b.shape[1] + sep] = (150, 150, 150)
    # 标签：左 OLD / 右 NEW（用英文避免 OpenCV 中文乱码）
    cv2.putText(canvas, "OLD", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 80, 40), 2)
    cv2.putText(canvas, "NEW", (b.shape[1] + sep + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 200), 2)
    return canvas


def _call_vlm(prompt, image_bgr, api_key, model, max_tokens=4096, timeout=180,
              base_url=DEFAULT_BASE_URL):
    """单次 VLM 图文调用，返回文本内容
    部分模型会忽略 enable_thinking=False，把输出写在 reasoning_content 里，
    故 content 为空时回退到 reasoning_content 参与解析。
    """
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("图像编码失败")
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    resp = requests.post(chat_completions_url(base_url), headers=headers,
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content


def _parse_region_json(content):
    """解析 VLM 返回的区域解读 JSON（容错处理）
    注意：content 可能来自 reasoning_content，JSON 多出现在末尾，故取最后一个匹配"""
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        text = code_match.group(1).strip()
    matches = re.findall(r'\{[\s\S]*?"is_real_change"[\s\S]*?\}', text)
    if not matches:
        matches = re.findall(r'\{[^{}]*\}', text)
    if not matches:
        raise ValueError(f"无法解析VLM返回: {content[:150]}")
    obj = None
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        raise ValueError(f"无法解析VLM返回: {content[:150]}")
    change_type = obj.get("change_type", "未知")
    if change_type not in CHANGE_TYPES:
        change_type = "未知"
    return {
        "is_real_change": bool(obj.get("is_real_change", True)),
        "change_type": change_type,
        "description": str(obj.get("description", ""))[:200],
    }


def _analyze_one_region(region, img_a, img_b, api_key, model, base_url):
    """解读单个差异区域"""
    crop_b = _crop_region(img_b, region["bbox"])
    crop_a = _crop_region(img_a, region["bbox"])
    combined = _make_labeled_pair(crop_b, crop_a)
    prompt = (
        "这是同一张工程图纸修改前后的同一局部区域：左侧标注OLD为旧版，右侧标注NEW为新版。"
        "请仔细对比两侧内容，找出新版相对旧版的变更。"
        "判断规则：\n"
        "- 尺寸变更：尺寸数字、公差、直径/半径标注的数值发生变化\n"
        "- 几何变更：轮廓线、孔、槽等几何形状发生增删或改变\n"
        "- 标注变更：文字注释、符号、表面粗糙度等标注的增删改\n"
        "- 视图移动：图形内容相同仅位置不同\n"
        "- 噪声：两侧内容实际一致，差异仅为线条粗细/渲染毛刺\n"
        "直接给出结论，不要输出分析过程。只返回一个JSON对象，不要有其他文字，格式：\n"
        '{"is_real_change": true, "change_type": "尺寸变更", "description": "孔径标注由25改为30"}\n'
        'change_type 只能取：尺寸变更/几何变更/标注变更/视图移动/噪声'
    )
    content = _call_vlm(prompt, combined, api_key, model, base_url=base_url)
    return _parse_region_json(content)


def analyze_regions(regions, img_a, img_b, api_key, model, max_workers=8, progress_callback=None,
                    base_url=DEFAULT_BASE_URL):
    """
    并行解读所有差异区域（原地更新 regions 各项的解读字段）
    img_a: 新图（BGR），img_b: 对齐后的旧图（BGR）
    单区域失败 → change_type="未知"，不阻断流程
    """
    if not regions:
        return regions

    done_count = 0

    def worker(region):
        try:
            return region, _analyze_one_region(region, img_a, img_b, api_key, model, base_url)
        except Exception as e:
            print(f"  区域{region['id']}解读失败: {e}")
            return region, {"is_real_change": True, "change_type": "未知", "description": ""}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, r) for r in regions]
        for fut in as_completed(futures):
            region, result = fut.result()
            region.update(result)
            done_count += 1
            if progress_callback:
                progress_callback(done_count, len(regions))
    return regions


def summarize_changes(regions, moved_views, api_key, model, base_url=DEFAULT_BASE_URL):
    """
    汇总各区域描述，生成 2~3 句变更摘要（纯文本调用，无图）
    失败返回空字符串
    """
    real_changes = [r for r in regions if r.get("is_real_change") and r.get("description")]
    if not real_changes and not moved_views:
        return ""
    lines = [f"{i+1}. [{r.get('change_type', '未知')}] {r['description']}"
             for i, r in enumerate(real_changes)]
    if moved_views:
        lines.append(f"另有{len(moved_views)}处视图整体位置移动（内容未变）")
    try:
        prompt = (
            "以下是一张工程图纸新旧版本对比检测出的变更明细，"
            "请用2~3句话概括本次图纸变更的核心内容，直接输出中文摘要文字，不要有其他说明：\n"
            + "\n".join(lines)
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        resp = requests.post(chat_completions_url(base_url), headers=headers,
                             json=payload, timeout=120)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # 摘要只信任正式 content，reasoning_content 是思考过程不可直接展示
        content = (msg.get("content") or "").strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content[:500]
    except Exception as e:
        print(f"  变更摘要生成失败: {e}")
        return ""

"""
视图分块与逐块配准模块：
  解决三视图/多视图图纸中视图位置整体移动时，全局单一配准失效的问题。

流程：
  segment_view_blocks  - 连通域分析切出各视图块（主动偏向过分割）
  match_blocks         - 块间 AKAZE 特征匹配（含先验硬约束 + RANSAC 校验）
  refine_matches       - 位移一致性聚类（碎片自愈）+ 离群可疑标记
  verify_match_vlm     - VLM 复核低置信度匹配
  compose_aligned_b    - 按块级变换重组对齐后的图B（未匹配块兜底原位贴回）
"""

import cv2
import numpy as np
import base64
import json
import re
import requests

from drawing_contrast.ai_config import DEFAULT_BASE_URL, chat_completions_url


def segment_view_blocks(gray_img, focus_mask=None, dilate_kernel=15, min_area_ratio=0.0005):
    """
    将图纸切分为视图块（连通域包围盒）
    gray_img: 灰度图（白底黑线）
    focus_mask: 可选，主体区域掩码（255=主体），排除边框/标题栏
    返回: [(x, y, w, h), ...] 按面积降序
    """
    h, w = gray_img.shape[:2]
    total_area = h * w

    # 二值化：提取线条（深色像素）
    _, binary = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)

    if focus_mask is not None:
        binary = cv2.bitwise_and(binary, focus_mask)

    # 膨胀粘连：核偏小，主动偏向过分割（碎片可被位移聚类自愈，粘连不可）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    blocks = []
    small = []
    min_area = total_area * min_area_ratio
    for i in range(1, num_labels):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area:
            small.append((x, y, bw, bh))
        else:
            blocks.append((x, y, bw, bh))

    # 小碎块并入最近的大块（按中心距离），无大块时丢弃
    for sx, sy, sw, sh in small:
        if not blocks:
            break
        scx, scy = sx + sw / 2, sy + sh / 2
        best_idx, best_dist = -1, None
        for idx, (bx, by, bw, bh) in enumerate(blocks):
            bcx, bcy = bx + bw / 2, by + bh / 2
            d = (scx - bcx) ** 2 + (scy - bcy) ** 2
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, idx
        # 距离过远（超过图幅对角线20%）则丢弃
        if best_dist is not None and best_dist < (0.2 * np.hypot(w, h)) ** 2:
            bx, by, bw, bh = blocks[best_idx]
            nx1, ny1 = min(bx, sx), min(by, sy)
            nx2, ny2 = max(bx + bw, sx + sw), max(by + bh, sy + sh)
            blocks[best_idx] = (nx1, ny1, nx2 - nx1, ny2 - ny1)

    blocks.sort(key=lambda b: b[2] * b[3], reverse=True)
    return blocks


def _crop_block(img, block, pad=10):
    """裁剪块区域（带边距），返回 (crop, (x1, y1))"""
    h, w = img.shape[:2]
    x, y, bw, bh = block
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y1:y2, x1:x2], (x1, y1)


def match_blocks(blocks_a, blocks_b, gray_a, gray_b, max_shift_ratio=0.4):
    """
    A、B 图视图块间的特征匹配
    返回匹配列表，每项为 dict:
      {block_a, block_b, warp, dx, dy, inliers, inlier_ratio, score}
    warp 为 2x3 仿射矩阵，将 B 块坐标（全图坐标系）映射到 A 坐标系
    """
    h, w = gray_a.shape[:2]
    max_shift = max(h, w) * max_shift_ratio
    detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 预计算每块的特征（坐标转回全图坐标系）
    def block_features(gray, blocks):
        feats = []
        for blk in blocks:
            crop, (ox, oy) = _crop_block(gray, blk)
            kp, desc = detector.detectAndCompute(crop, None)
            pts = np.float32([(k.pt[0] + ox, k.pt[1] + oy) for k in kp]) if kp else np.zeros((0, 2), np.float32)
            feats.append((pts, desc))
        return feats

    feats_a = block_features(gray_a, blocks_a)
    feats_b = block_features(gray_b, blocks_b)

    candidates = []
    for ia, blk_a in enumerate(blocks_a):
        pts_a_all, desc_a = feats_a[ia]
        if desc_a is None or len(pts_a_all) < 8:
            continue
        aw, ah = blk_a[2], blk_a[3]
        for ib, blk_b in enumerate(blocks_b):
            bw, bh = blk_b[2], blk_b[3]
            # 先验硬约束1: 面积比 [0.5, 2]
            area_ratio = (aw * ah) / max(1, bw * bh)
            if not (0.5 <= area_ratio <= 2.0):
                continue
            # 先验硬约束2: 纵横比接近
            ar_a, ar_b = aw / max(1, ah), bw / max(1, bh)
            if not (0.5 <= ar_a / max(1e-6, ar_b) <= 2.0):
                continue
            # 先验硬约束3: 中心位移上限
            dcx = (blk_a[0] + aw / 2) - (blk_b[0] + bw / 2)
            dcy = (blk_a[1] + ah / 2) - (blk_b[1] + bh / 2)
            if np.hypot(dcx, dcy) > max_shift:
                continue

            pts_b_all, desc_b = feats_b[ib]
            if desc_b is None or len(pts_b_all) < 8:
                continue

            knn = matcher.knnMatch(desc_b, desc_a, k=2)
            good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                continue

            src = np.float32([pts_b_all[m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([pts_a_all[m.trainIdx] for m in good]).reshape(-1, 1, 2)
            M, inlier_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                         ransacReprojThreshold=3.0)
            if M is None:
                continue
            inliers = int(np.sum(inlier_mask))
            inlier_ratio = inliers / len(good)
            if inliers < 6:
                continue
            # 缩放合理性检查（同一零件不同版本缩放接近1.0）
            scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
            if not (0.8 <= scale <= 1.25):
                continue
            candidates.append({
                "ia": ia, "ib": ib,
                "block_a": blk_a, "block_b": blk_b,
                "warp": M.astype(np.float32),
                "dx": float(M[0, 2]), "dy": float(M[1, 2]),
                "inliers": inliers, "inlier_ratio": inlier_ratio,
                "score": inliers * inlier_ratio,
            })

    # 贪心一对一分配（按得分降序）
    candidates.sort(key=lambda c: c["score"], reverse=True)
    used_a, used_b = set(), set()
    matches = []
    for c in candidates:
        if c["ia"] in used_a or c["ib"] in used_b:
            continue
        used_a.add(c["ia"])
        used_b.add(c["ib"])
        matches.append(c)

    unmatched_a = [blocks_a[i] for i in range(len(blocks_a)) if i not in used_a]
    unmatched_b = [blocks_b[i] for i in range(len(blocks_b)) if i not in used_b]
    return matches, unmatched_a, unmatched_b


def refine_matches(matches, cluster_dist=20.0, suspect_inlier_ratio=0.35):
    """
    位移一致性聚类：同一视图的碎片位移必然一致，天然聚成一簇。
    - 同簇匹配共用组内中位数变换（碎片自愈）
    - 孤立簇且内点率低 → suspect=True 待复核
    """
    if not matches:
        return matches

    # 简单贪心聚类位移向量
    clusters = []  # 每项: [indices]
    for i, m in enumerate(matches):
        placed = False
        for cl in clusters:
            ref = matches[cl[0]]
            if np.hypot(m["dx"] - ref["dx"], m["dy"] - ref["dy"]) < cluster_dist:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    for cl in clusters:
        if len(cl) >= 2:
            # 同簇共用中位数变换（对碎片更稳）
            warps = np.stack([matches[i]["warp"] for i in cl])
            median_warp = np.median(warps, axis=0).astype(np.float32)
            for i in cl:
                matches[i]["warp"] = median_warp
                matches[i]["dx"] = float(median_warp[0, 2])
                matches[i]["dy"] = float(median_warp[1, 2])
                matches[i]["suspect"] = False
                matches[i]["cluster_size"] = len(cl)
        else:
            i = cl[0]
            matches[i]["cluster_size"] = 1
            # 孤立位移簇 + 内点率低 → 可疑
            matches[i]["suspect"] = matches[i]["inlier_ratio"] < suspect_inlier_ratio
    return matches


def _ink_dice(gray_a, gray_b, block_b, warp):
    """
    评估位移假设：把 B 块按 warp 映射到 A 坐标系后，计算两侧墨迹的 Dice 相似度
    返回 0~1，越高说明该位移下内容越一致
    """
    h, w = gray_a.shape[:2]
    x, y, bw, bh = block_b
    crop_b = gray_b[y:y + bh, x:x + bw]
    # 块左上角经 warp 映射后的位置（近似取平移分量，块内旋转很小）
    nx = int(round(warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]))
    ny = int(round(warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]))
    if nx < -bw or ny < -bh or nx >= w or ny >= h:
        return 0.0
    x1, y1 = max(0, nx), max(0, ny)
    x2, y2 = min(w, nx + bw), min(h, ny + bh)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0.0
    crop_a = gray_a[y1:y2, x1:x2]
    crop_b2 = crop_b[y1 - ny:y2 - ny, x1 - nx:x2 - nx]
    ink_a = crop_a < 180
    ink_b = crop_b2 < 180
    total = int(ink_a.sum()) + int(ink_b.sum())
    if total == 0:
        return 0.0
    return 2.0 * int((ink_a & ink_b).sum()) / total


def rescue_unmatched(matches, unmatched_a, unmatched_b, gray_a, gray_b, dice_thresh=0.5):
    """
    碎片救援：未匹配的 B 块用已知簇位移（+零位移）逐一试探，
    墨迹 Dice 达标则认领该位移，避免被移动视图的碎片被误报为删除+新增。
    同时清理被救援位置覆盖的 A 侧未匹配块。
    返回: (matches, unmatched_a, unmatched_b)
    """
    if not unmatched_b:
        return matches, unmatched_a, unmatched_b

    # 候选位移：各匹配簇的 warp（去重）+ 零位移
    candidates = [np.float32([[1, 0, 0], [0, 1, 0]])]
    seen = set()
    for m in matches:
        key = (round(m["dx"] / 10), round(m["dy"] / 10))
        if key not in seen:
            seen.add(key)
            candidates.append(m["warp"])

    still_unmatched_b = []
    rescued = []
    for blk in unmatched_b:
        best_warp, best_score = None, 0.0
        for warp in candidates:
            score = _ink_dice(gray_a, gray_b, blk, warp)
            if score > best_score:
                best_score, best_warp = score, warp
        if best_warp is not None and best_score >= dice_thresh:
            rescued.append({
                "block_a": blk, "block_b": blk,
                "warp": best_warp,
                "dx": float(best_warp[0, 2]), "dy": float(best_warp[1, 2]),
                "inliers": 0, "inlier_ratio": best_score,
                "score": 0.0, "suspect": False, "cluster_size": 1,
                "rescued": True,
            })
            print(f"  碎片救援: 块{blk} 采用位移({best_warp[0, 2]:.0f}, {best_warp[1, 2]:.0f}), Dice={best_score:.2f}")
        else:
            still_unmatched_b.append(blk)

    if rescued:
        matches = matches + rescued
        # 清理被救援块目标位置覆盖的 A 侧未匹配块（内容已有着落）
        def overlap_ratio(box1, box2):
            x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            return inter / max(1, box1[2] * box1[3])
        remaining_a = []
        for blk_a in unmatched_a:
            covered = False
            for m in rescued:
                x, y, bw, bh = m["block_b"]
                warped_box = (int(x + m["dx"]), int(y + m["dy"]), bw, bh)
                if overlap_ratio(blk_a, warped_box) > 0.5:
                    covered = True
                    break
            if not covered:
                remaining_a.append(blk_a)
        unmatched_a = remaining_a

    return matches, unmatched_a, still_unmatched_b


def _make_side_by_side(crop_a, crop_b, target_long=512):
    """两块图左右拼图（等比缩放到相近尺寸，中间画灰色分隔线）"""
    def resize_to(img, long_side):
        h, w = img.shape[:2]
        s = long_side / max(h, w)
        return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))

    a = resize_to(crop_a, target_long)
    b = resize_to(crop_b, target_long)
    h = max(a.shape[0], b.shape[0])
    sep = 6
    canvas = np.full((h, a.shape[1] + sep + b.shape[1], 3), 255, dtype=np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1] + sep:] = b
    canvas[:, a.shape[1]:a.shape[1] + sep] = (160, 160, 160)
    return canvas


def verify_match_vlm(crop_a, crop_b, api_key, model, timeout=60,
                     base_url=DEFAULT_BASE_URL):
    """
    VLM 复核：判断两个视图块是否为同一图形内容
    返回: (is_same: bool | None)，None 表示调用失败
    """
    try:
        combined = _make_side_by_side(crop_a, crop_b)
        ok, buf = cv2.imencode(".png", combined)
        if not ok:
            return None
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        prompt = (
            "这张图由左右两个工程图纸局部拼接而成（中间有灰色分隔线）。"
            "请判断左右两部分是否为【同一个视图/图形内容】（允许存在少量尺寸标注或细节修改，"
            "只要主体图形一致即视为相同）。"
            '只返回JSON，格式：{"is_same": true} 或 {"is_same": false}，不要其他文字。'
        )
        url = chat_completions_url(base_url)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}],
            "max_tokens": 2048,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # 部分模型忽略 enable_thinking=False，输出写在 reasoning_content 里
        content = msg.get("content") or ""
        if not content.strip():
            content = msg.get("reasoning_content") or ""
        text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 取最后一个匹配（思考文本中结论在末尾）
        found = re.findall(r'\{[^{}]*"is_same"[^{}]*\}', text)
        if found:
            return bool(json.loads(found[-1])["is_same"])
        # 兜底：文本关键词
        if "true" in text.lower():
            return True
        if "false" in text.lower():
            return False
        return None
    except Exception as e:
        print(f"  VLM匹配复核失败: {e}")
        return None


def compose_aligned_b(img_b_global, matches, unmatched_b, shape, focus_mask=None):
    """
    重组对齐后的图B：
    - 每个匹配块按各自 warp 变换后贴到 A 坐标系
    - 未匹配的 B 块按全局配准位置原样贴回（兜底不劣化）
    - 主体区域外的内容（图框、边缘视图等）从原图B原样保留，避免被裁成白底
    img_b_global: 已做过全局配准的图B（BGR）
    返回: 重组后的 aligned_b（BGR，白底）
    """
    h, w = shape[:2]
    if focus_mask is not None and focus_mask.shape[:2] == (h, w):
        # 主体区域外直接以原图B打底，块级变换只影响主体区域内
        canvas = img_b_global.copy()
        m3 = cv2.merge([focus_mask] * 3)
        canvas = np.where(m3 > 0, 255, canvas).astype(np.uint8)
    else:
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    def paste_mask(dst, src, mask):
        """按掩码把 src 中的线条像素贴到 dst（取较暗值，避免白边覆盖已有线条）"""
        region = mask > 0
        dst[region] = np.minimum(dst[region], src[region])

    for m in matches:
        x, y, bw, bh = m["block_b"]
        # 块区域掩码（带边距，保证边界线条完整）
        pad = 10
        block_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        block_mask[y1:y2, x1:x2] = 255

        warped = cv2.warpAffine(img_b_global, m["warp"], (w, h),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        warped_mask = cv2.warpAffine(block_mask, m["warp"], (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        paste_mask(canvas, warped, warped_mask)

    for x, y, bw, bh in unmatched_b:
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        paste_mask(canvas, img_b_global, mask)

    return canvas


def block_level_align(img_a, img_b_global, focus_mask=None, api_key=None, model=None,
                      report=None, base_url=DEFAULT_BASE_URL):
    """
    视图分块逐块配准的总入口

    img_a: 图A（BGR）
    img_b_global: 已全局配准的图B（BGR）
    api_key/model: 可选，用于 VLM 复核可疑匹配
    report: 可选日志回调 report(msg)

    返回: (aligned_b, block_info)
      aligned_b: 逐块重组后的图B
      block_info: dict，含 moved_views / view_stats / low_confidence
    """
    def log(msg):
        if report:
            report(msg)
        print(msg)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_global, cv2.COLOR_BGR2GRAY)

    blocks_a = segment_view_blocks(gray_a, focus_mask)
    blocks_b = segment_view_blocks(gray_b, focus_mask)
    log(f"  视图分块: A={len(blocks_a)}块, B={len(blocks_b)}块")

    if not blocks_a or not blocks_b:
        log("  分块失败，回退到全局配准结果")
        return img_b_global, {
            "moved_views": [],
            "view_stats": {"matched": 0, "added": len(blocks_a), "removed": len(blocks_b), "suspect": 0},
            "low_confidence": False,
        }

    matches, unmatched_a, unmatched_b = match_blocks(blocks_a, blocks_b, gray_a, gray_b)
    matches = refine_matches(matches)
    log(f"  块匹配: {len(matches)}对匹配, A侧未匹配{len(unmatched_a)}块, B侧未匹配{len(unmatched_b)}块")

    # 碎片救援：用已知簇位移试探未匹配块，避免移动视图的碎片产生误报
    matches, unmatched_a, unmatched_b = rescue_unmatched(
        matches, unmatched_a, unmatched_b, gray_a, gray_b)

    # VLM 复核可疑匹配
    low_confidence = False
    suspects = [m for m in matches if m.get("suspect")]
    if suspects:
        log(f"  发现{len(suspects)}对可疑匹配（孤立位移+低内点率）")
        for m in suspects:
            if api_key and model:
                crop_a, _ = _crop_block(img_a, m["block_a"])
                crop_b, _ = _crop_block(img_b_global, m["block_b"])
                is_same = verify_match_vlm(crop_a, crop_b, api_key, model, base_url=base_url)
                if is_same is False:
                    # 撤销匹配，降级为新增/删除视图
                    m["rejected"] = True
                    unmatched_a.append(m["block_a"])
                    unmatched_b.append(m["block_b"])
                    log("  VLM判定为不同内容，撤销该匹配")
                elif is_same is None:
                    low_confidence = True
                    log("  VLM复核失败，保留匹配（低置信度）")
            else:
                low_confidence = True
        matches = [m for m in matches if not m.get("rejected")]

    aligned_b = compose_aligned_b(img_b_global, matches, unmatched_b, img_a.shape, focus_mask)

    # 统计移动视图：位移显著（>15px）的匹配（去重：按簇合并后的位移分组取包围盒）
    moved_views = []
    seen_clusters = {}
    for m in matches:
        shift = np.hypot(m["dx"], m["dy"])
        if shift <= 15:
            continue
        key = (round(m["dx"] / 10), round(m["dy"] / 10))
        xa, ya, wa, ha = m["block_a"]
        xb, yb, wb, hb = m["block_b"]
        if key in seen_clusters:
            mv = seen_clusters[key]
            # 扩展包围盒
            ax1, ay1 = min(mv["bbox_a"][0], xa), min(mv["bbox_a"][1], ya)
            ax2 = max(mv["bbox_a"][0] + mv["bbox_a"][2], xa + wa)
            ay2 = max(mv["bbox_a"][1] + mv["bbox_a"][3], ya + ha)
            mv["bbox_a"] = [ax1, ay1, ax2 - ax1, ay2 - ay1]
            bx1, by1 = min(mv["bbox_b"][0], xb), min(mv["bbox_b"][1], yb)
            bx2 = max(mv["bbox_b"][0] + mv["bbox_b"][2], xb + wb)
            by2 = max(mv["bbox_b"][1] + mv["bbox_b"][3], yb + hb)
            mv["bbox_b"] = [bx1, by1, bx2 - bx1, by2 - by1]
        else:
            seen_clusters[key] = {
                "bbox_a": [int(xa), int(ya), int(wa), int(ha)],
                "bbox_b": [int(xb), int(yb), int(wb), int(hb)],
                "dx": round(m["dx"], 1), "dy": round(m["dy"], 1),
            }
    moved_views = list(seen_clusters.values())
    if moved_views:
        log(f"  检测到{len(moved_views)}处视图移动")

    block_info = {
        "moved_views": moved_views,
        "view_stats": {
            "matched": len(matches),
            "added": len(unmatched_a),
            "removed": len(unmatched_b),
            "suspect": len(suspects),
        },
        "low_confidence": low_confidence,
    }
    return aligned_b, block_info

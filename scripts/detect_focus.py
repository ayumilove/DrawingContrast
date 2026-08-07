import fitz
import base64
import os
import sys
import requests
import json
from pathlib import Path

from drawing_contrast import paths
from drawing_contrast.ai_config import DEFAULT_BASE_URL, chat_completions_url


def pdf_page_to_base64(pdf_path, page_num=0, zoom=2):
    """将PDF指定页面转为base64编码的PNG图像"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def detect_focus_area(image_base64, api_key, model="Qwen/Qwen3-VL-32B-Instruct",
                      base_url=DEFAULT_BASE_URL):
    """
    调用多模态大模型，识别图纸主体区域边界框
    返回: (left, top, right, bottom) 均为 0~1 的浮点数比例
    """
    # OpenAI 兼容 API（默认硅基流动 SiliconFlow）
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
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1
    }

    response = requests.post(url, headers=headers, json=payload, timeout=180)
    print(f"HTTP状态码: {response.status_code}")
    response.raise_for_status()
    result = response.json()
    print(f"完整响应:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")

    # 提取模型返回的文本
    content = result["choices"][0]["message"]["content"]
    print(f"模型原始返回:\n{content}\n")

    # 尝试从返回内容中解析JSON
    # 模型可能会用 ```json ... ``` 包裹，也可能直接返回JSON
    import re
    json_match = re.search(r'\{[^}]*"left"[^}]*\}', content, re.DOTALL)
    if json_match:
        coords = json.loads(json_match.group())
        return (
            float(coords["left"]),
            float(coords["top"]),
            float(coords["right"]),
            float(coords["bottom"])
        )
    else:
        raise ValueError(f"无法从模型返回中解析坐标: {content}")


if __name__ == "__main__":
    base_dir = paths.program_dir()
    # 支持命令行指定 PDF： uv run scripts/detect_focus.py <pdf路径>
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        raise SystemExit("用法: uv run scripts/detect_focus.py <pdf文件路径>")
    if not pdf_path.exists():
        raise SystemExit(f"文件不存在: {pdf_path}")

    # 从环境变量读取API Key（避免硬编码提交到仓库）
    api_key = os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("AI_API_KEY", "")
    if not api_key:
        raise SystemExit("请先设置环境变量 SILICONFLOW_API_KEY（可选 AI_BASE_URL / AI_MODEL）")
    base_url = os.environ.get("AI_BASE_URL", DEFAULT_BASE_URL)

    print("正在将PDF转为图像...")
    image_b64 = pdf_page_to_base64(str(pdf_path), page_num=0, zoom=2)
    print(f"图像base64长度: {len(image_b64)}")

    print("正在调用多模态模型识别主体区域...")
    try:
        left, top, right, bottom = detect_focus_area(image_b64, api_key, base_url=base_url)
        print(f"\n识别到的主体区域: left={left:.3f}, top={top:.3f}, right={right:.3f}, bottom={bottom:.3f}")

        # 保存结果供对比脚本使用
        focus_result = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom
        }
        with open(base_dir / "focus_area.json", "w", encoding="utf-8") as f:
            json.dump(focus_result, f, indent=2, ensure_ascii=False)
        print("已保存到 focus_area.json")

    except Exception as e:
        print(f"调用失败: {e}")

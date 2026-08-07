"""
生成应用图标 - 工程图纸差异对比工具
用法: uv run scripts/generate_icon.py
"""
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path


def create_icon():
    """生成一个现代风格的图纸对比图标"""
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 背景：圆角矩形，蓝色渐变感
    # 先画一个圆形底色
    padding = 16
    bg_rect = [padding, padding, size - padding, size - padding]

    # 圆角背景
    radius = 48
    draw.rounded_rectangle(bg_rect, radius=radius, fill=(37, 99, 235))  # #2563eb

    # 左侧文档图标（旧图 - 白色半透明）
    doc_w, doc_h = 70, 90
    # 左文档
    lx, ly = 50, 75
    draw.rounded_rectangle(
        [lx, ly, lx + doc_w, ly + doc_h],
        radius=8, fill=(255, 255, 255, 180)
    )
    # 左文档上的"线条"
    for i in range(4):
        y = ly + 18 + i * 16
        draw.line([(lx + 10, y), (lx + doc_w - 10, y)], fill=(37, 99, 235, 120), width=3)

    # 右侧文档图标（新图 - 白色实心）
    rx, ry = 136, 90
    draw.rounded_rectangle(
        [rx, ry, rx + doc_w, ry + doc_h],
        radius=8, fill=(255, 255, 255, 240)
    )
    # 右文档上的"线条"（有变化的用红色）
    for i in range(4):
        y = ry + 18 + i * 16
        color = (239, 68, 68, 200) if i == 1 or i == 3 else (37, 99, 235, 120)
        draw.line([(rx + 10, y), (rx + doc_w - 10, y)], fill=color, width=3)

    # 中间的对比箭头/差异标记
    # 画一个圆形对比符号
    cx, cy = 128, 128
    circle_r = 18
    draw.ellipse(
        [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
        fill=(255, 255, 255)
    )
    # 在圆里画 "≠" 符号
    draw.line([(cx - 8, cy - 4), (cx + 8, cy - 4)], fill=(239, 68, 68), width=3)
    draw.line([(cx - 8, cy + 4), (cx + 8, cy + 4)], fill=(239, 68, 68), width=3)
    # 斜线
    draw.line([(cx + 6, cy - 10), (cx - 6, cy + 10)], fill=(239, 68, 68), width=2)

    # 保存为多尺寸 .ico（输出到包资源目录）
    icon_path = str(Path(__file__).resolve().parents[1] / 'src' / 'drawing_contrast' / 'resources' / 'app_icon.ico')
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icons = []
    for s in sizes:
        resized = img.resize(s, Image.Resampling.LANCZOS)
        icons.append(resized)

    icons[0].save(
        icon_path, format='ICO',
        sizes=[(s.width, s.height) for s in icons],
        append_images=icons[1:]
    )
    print(f"图标已生成: {icon_path}")
    return icon_path


if __name__ == "__main__":
    create_icon()

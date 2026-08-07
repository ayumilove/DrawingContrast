"""
统一路径解析模块。

- resources_dir：随包分发的只读资源（web 页面、图标）。打包后在 _MEIPASS，开发时在包目录
- program_dir：运行时产物写入目录（config.json、app.log、web 输出）。
  打包后为 EXE 旁目录，开发时为项目根目录
"""

import sys
import shutil
from pathlib import Path


def is_frozen() -> bool:
    """是否为 PyInstaller 打包环境"""
    return hasattr(sys, "_MEIPASS")


def package_dir() -> Path:
    """包所在目录"""
    return Path(__file__).parent


def resources_dir() -> Path:
    """只读资源目录：打包后 _MEIPASS/resources，开发时 包目录/resources"""
    if is_frozen():
        return Path(sys._MEIPASS) / "resources"
    return package_dir() / "resources"


def program_dir() -> Path:
    """程序目录（运行时产物写入位置）：打包后 EXE 旁，开发时 项目根目录"""
    if is_frozen():
        return Path(sys.executable).parent
    # src/drawing_contrast/paths.py → 上两级即项目根目录
    return package_dir().parents[1]


def web_dir() -> Path:
    """Web 运行时目录：index.html 页面与 static/ 对比产物"""
    return program_dir() / "web"


def sync_web() -> Path:
    """将只读 web 页面从包资源同步到运行时目录（仅覆盖页面文件，不触碰 static/）"""
    src = resources_dir() / "web" / "index.html"
    dst_dir = web_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "index.html"
    shutil.copyfile(src, dst)
    return dst

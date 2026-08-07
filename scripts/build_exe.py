"""
PyInstaller 打包脚本
用法: uv run scripts/build_exe.py
"""
import PyInstaller.__main__
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
pkg_dir = base_dir / "src" / "drawing_contrast"

PyInstaller.__main__.run([
    str(pkg_dir / "app.py"),
    "--onefile",
    "--windowed",
    "--name=DrawingContrast",
    f"--icon={pkg_dir / 'resources' / 'app_icon.ico'}",
    f"--add-data={pkg_dir / 'resources'};resources",
    "--hidden-import=PyQt6.QtWebEngineWidgets",
    "--hidden-import=PyQt6.QtWebEngineCore",
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=fitz",
    "--hidden-import=requests",
    "--hidden-import=drawing_contrast",
    "--hidden-import=drawing_contrast.core.generate_diff_layers",
    "--hidden-import=drawing_contrast.core.view_blocks",
    "--hidden-import=drawing_contrast.core.vlm_analyzer",
    "--hidden-import=drawing_contrast.integrations.ftp_downloader",
    "--hidden-import=drawing_contrast.integrations.logbot",
    "--hidden-import=drawing_contrast.integrations.updater",
    f"--distpath={base_dir / 'dist'}",
    f"--workpath={base_dir / 'build'}",
    f"--specpath={base_dir}",
    "--noconfirm",
])

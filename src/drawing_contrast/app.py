"""
工程图纸差异对比工具 - 桌面 GUI 应用
双击启动后可选择两个 PDF 图纸（变更前/变更后），自动完成对比并在内嵌浏览器中展示交互结果。
"""

import sys
import os
import time
import threading
import socket
import http.server
import json
import tempfile
import logging
import requests
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QLineEdit, QMessageBox, QStatusBar, QSplashScreen,
    QDialog, QFormLayout, QInputDialog, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QPainter, QColor, QLinearGradient, QPen,
    QDesktopServices
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile

from drawing_contrast import ai_config, paths
from drawing_contrast import __version__ as APP_VERSION
from drawing_contrast.core.generate_diff_layers import run_comparison
from drawing_contrast.integrations.ftp_downloader import FtpDownloader, FtpDownloadError
from drawing_contrast.integrations.logbot import LogBotSdk
from drawing_contrast.integrations.updater import fetch_latest_release, parse_version

# 应用版本（与 pyproject.toml 的 version 字段保持一致，取自包定义）


# ---------- 日志系统 ----------

def setup_logger():
    """配置文件日志，写入程序目录（打包后 EXE 旁，开发时项目根目录）下的 app.log"""
    log_dir = paths.program_dir()
    log_file = log_dir / "app.log"

    logger = logging.getLogger("DrawingContrast")
    logger.setLevel(logging.DEBUG)

    # 文件 handler（追加模式，UTF-8 编码）
    fh = logging.FileHandler(str(log_file), encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # 同时输出到控制台（开发调试用）
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    logger.info(f"=" * 50)
    logger.info(f"程序启动 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"日志文件: {log_file}")
    return logger


# 全局 logger（在 main() 中初始化）
log = None
logbot = None  # 全局 LogBot SDK 实例（仅当配置了 log_url 时启用）


# ---------- 内置 HTTP 服务器 ----------

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    """静态文件 Handler，指向 web 目录"""

    def __init__(self, *args, web_dir=None, **kwargs):
        self._web_dir = web_dir
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format, *args):
        pass  # 静默日志


def find_free_port():
    """获取可用随机端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class WebServerThread(threading.Thread):
    """后台 HTTP 服务器线程"""

    def __init__(self, web_dir, port):
        super().__init__(daemon=True)
        self.web_dir = web_dir
        self.port = port
        self.server = None

    def run(self):
        handler = lambda *args, **kwargs: StaticHandler(
            *args, web_dir=str(self.web_dir), **kwargs
        )
        self.server = http.server.HTTPServer(('127.0.0.1', self.port), handler)
        self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()


# ---------- 后台对比工作线程 ----------

class ComparisonWorker(QThread):
    """后台执行图纸对比"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, pdf_a, pdf_b, output_dir, api_key, use_ai, model, base_url,
                 enhance_lines=True, annotate_changes=True):
        super().__init__()
        self.pdf_a = pdf_a
        self.pdf_b = pdf_b
        self.output_dir = output_dir
        self.api_key = api_key
        self.use_ai = use_ai
        self.model = model
        self.base_url = base_url
        self.enhance_lines = enhance_lines
        self.annotate_changes = annotate_changes

    def run(self):
        try:
            meta = run_comparison(
                self.pdf_a, self.pdf_b, self.output_dir,
                api_key=self.api_key,
                use_ai=self.use_ai,
                model=self.model,
                progress_callback=self._on_progress,
                base_url=self.base_url,
                enhance_lines=self.enhance_lines,
                annotate_changes=self.annotate_changes
            )
            self.finished.emit(meta)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, pct, msg):
        self.progress.emit(pct, msg)


# ---------- 后台更新检查线程 ----------

class UpdateCheckThread(QThread):
    """后台检查 GitHub Release 新版本；失败静默降级，不影响主流程"""
    checked = pyqtSignal(object)  # 最新 Release 信息 dict，不可用时为 None

    def run(self):
        self.checked.emit(fetch_latest_release())


# ---------- AI 设置对话框 ----------

class SettingsDialog(QDialog):
    """设置对话框：AI 配置 + FTP（可选）+ 日志上报（可选）
    Key 经 DPAPI 加密本地保存；支持分享码 / 明文配置文件两种方式分发"""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        # ── AI 设置 ──
        grp_ai = QGroupBox("AI 设置（可选，未配置时降级为纯 CV 对比）")
        form_ai = QFormLayout(grp_ai)
        form_ai.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_api_key = QLineEdit()
        self.edit_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_api_key.setPlaceholderText("输入 API Key（支持任意 OpenAI 兼容服务）")
        self.edit_api_key.setText(cfg.get("api_key", ""))
        self.edit_base_url = QLineEdit()
        self.edit_base_url.setPlaceholderText(ai_config.DEFAULT_BASE_URL)
        self.edit_base_url.setText(cfg.get("base_url", ai_config.DEFAULT_BASE_URL))
        self.edit_model = QLineEdit()
        self.edit_model.setPlaceholderText(ai_config.DEFAULT_MODEL)
        self.edit_model.setText(cfg.get("model", ai_config.DEFAULT_MODEL))
        form_ai.addRow("API Key", self.edit_api_key)
        form_ai.addRow("Base URL", self.edit_base_url)
        form_ai.addRow("模型名称", self.edit_model)
        layout.addWidget(grp_ai)

        # ── 对比选项 ──
        grp_view = QGroupBox("对比选项")
        form_view = QFormLayout(grp_view)
        form_view.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.chk_enhance = QCheckBox("PDF 转图片时加粗/加深细线条（推荐开启）")
        self.chk_enhance.setChecked(bool(cfg.get("enhance_lines", True)))
        self.chk_annotate = QCheckBox("框选差异区域并生成变更清单与文字描述（关闭后仅显示红/蓝差异）")
        self.chk_annotate.setChecked(bool(cfg.get("annotate_changes", True)))
        form_view.addRow("线条增强", self.chk_enhance)
        form_view.addRow("差异标注", self.chk_annotate)
        layout.addWidget(grp_view)

        # ── FTP 设置 ──
        grp_ftp = QGroupBox("FTP 设置（可选，配置后启用零部件编号下载对比）")
        form_ftp = QFormLayout(grp_ftp)
        form_ftp.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_ftp_host = QLineEdit()
        self.edit_ftp_host.setPlaceholderText("如 192.168.1.100，留空则不启用 FTP 功能")
        self.edit_ftp_host.setText(cfg.get("ftp_host", ""))
        self.edit_ftp_port = QLineEdit()
        self.edit_ftp_port.setPlaceholderText("21")
        self.edit_ftp_port.setText(str(cfg.get("ftp_port", 21)))
        self.edit_ftp_user = QLineEdit()
        self.edit_ftp_user.setText(cfg.get("ftp_user", ""))
        self.edit_ftp_pass = QLineEdit()
        self.edit_ftp_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_ftp_pass.setText(cfg.get("ftp_pass", ""))
        self.edit_ftp_base_dir = QLineEdit()
        self.edit_ftp_base_dir.setPlaceholderText("可选，留空表示从根目录查找")
        self.edit_ftp_base_dir.setText(cfg.get("ftp_base_dir", ""))
        form_ftp.addRow("主机地址", self.edit_ftp_host)
        form_ftp.addRow("端口", self.edit_ftp_port)
        form_ftp.addRow("用户名", self.edit_ftp_user)
        form_ftp.addRow("密码", self.edit_ftp_pass)
        form_ftp.addRow("基础目录", self.edit_ftp_base_dir)
        layout.addWidget(grp_ftp)

        # ── 日志上报设置 ──
        grp_log = QGroupBox("日志上报设置（可选，未配置则不上报）")
        form_log = QFormLayout(grp_log)
        form_log.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_log_url = QLineEdit()
        self.edit_log_url.setPlaceholderText("如 http://your-log-service/api/v1/logs，留空则不上报")
        self.edit_log_url.setText(cfg.get("log_url", ""))
        form_log.addRow("上报地址", self.edit_log_url)
        layout.addWidget(grp_log)

        tip = QLabel(
            "保存后 Key 经 Windows DPAPI 加密存于程序同目录（仅本机当前用户可解密）。"
            "分享给他人：「导出分享码」粘贴给对方导入；或「导出配置文件」，"
            "对方将文件放到程序同目录即可自动生效（含 FTP/日志配置）。"
        )
        tip.setStyleSheet("color:#64748b; font-size:11px;")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        # 底部按钮行：测试/分享/导出 + 取消/保存
        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("测试连接")
        self.btn_test.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_test.setToolTip("验证 AI 的 Base URL 与 Key 是否有效")
        self.btn_test.clicked.connect(self._test_connection)
        btn_row.addWidget(self.btn_test)
        self.btn_export = QPushButton("导出分享码")
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.setToolTip("把全部配置（含 FTP/日志）打包成分享码，复制给其他用户导入使用")
        self.btn_export.clicked.connect(self._export_share_code)
        btn_row.addWidget(self.btn_export)
        self.btn_import = QPushButton("导入分享码")
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import.setToolTip("粘贴他人分享的配置码，自动填充表单")
        self.btn_import.clicked.connect(self._import_share_code)
        btn_row.addWidget(self.btn_import)
        self.btn_export_file = QPushButton("导出配置文件")
        self.btn_export_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_file.setToolTip("导出明文配置文件供分发，对方放到程序同目录即可无感生效")
        self.btn_export_file.clicked.connect(self._export_config_file)
        btn_row.addWidget(self.btn_export_file)
        self.lbl_test = QLabel("")
        btn_row.addWidget(self.lbl_test, stretch=1)
        btn_cancel = QPushButton("取消")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("保存")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def get_config(self) -> dict:
        try:
            ftp_port = int(self.edit_ftp_port.text().strip() or 21)
        except ValueError:
            ftp_port = 21
        return {
            "api_key": self.edit_api_key.text().strip(),
            "base_url": self.edit_base_url.text().strip() or ai_config.DEFAULT_BASE_URL,
            "model": self.edit_model.text().strip() or ai_config.DEFAULT_MODEL,
            "enable_ai": True,
            "enhance_lines": self.chk_enhance.isChecked(),
            "annotate_changes": self.chk_annotate.isChecked(),
            "ftp_host": self.edit_ftp_host.text().strip(),
            "ftp_port": ftp_port,
            "ftp_user": self.edit_ftp_user.text().strip(),
            "ftp_pass": self.edit_ftp_pass.text(),
            "ftp_base_dir": self.edit_ftp_base_dir.text().strip(),
            "log_url": self.edit_log_url.text().strip(),
        }

    def _test_connection(self):
        """调用 /models 接口验证 Base URL 与 Key 是否有效"""
        cfg = self.get_config()
        self.lbl_test.setStyleSheet("color:#64748b;")
        self.lbl_test.setText("正在测试…")
        QApplication.processEvents()
        try:
            url = ai_config.normalize_base_url(cfg["base_url"]) + "/models"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {cfg['api_key']}"},
                timeout=10,
            )
            resp.raise_for_status()
            self.lbl_test.setStyleSheet("color:#16a34a;")
            self.lbl_test.setText("连接成功 ✓")
        except Exception as e:
            self.lbl_test.setStyleSheet("color:#dc2626;")
            self.lbl_test.setText(f"连接失败: {e}")

    def _export_share_code(self):
        """导出分享码到剪贴板：把当前表单配置打包，供其他用户导入"""
        cfg = self.get_config()
        if not cfg["api_key"]:
            QMessageBox.warning(self, "提示", "请先填写 API Key 再导出分享码")
            return
        token = ai_config.encode_share_token(cfg)
        QApplication.clipboard().setText(token)
        QMessageBox.information(
            self, "已导出",
            "分享码已复制到剪贴板，直接粘贴发送给对方即可。\n\n"
            "注意：分享码内含明文 Key，请仅通过可信渠道分享给信任的人。\n"
            "对方收到后在「AI 设置」中点「导入分享码」即可使用。"
        )

    def _import_share_code(self):
        """导入分享码：粘贴后解析并填充表单，需点「保存」才落盘"""
        token, ok = QInputDialog.getText(
            self, "导入分享码", "请粘贴收到的分享码（DC1: 开头）：",
            QLineEdit.EchoMode.Normal, QApplication.clipboard().text()
        )
        if not ok:
            return
        try:
            cfg = ai_config.decode_share_token(token)
        except ValueError as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return
        self.edit_api_key.setText(cfg.get("api_key", ""))
        self.edit_base_url.setText(cfg.get("base_url", ai_config.DEFAULT_BASE_URL))
        self.edit_model.setText(cfg.get("model", ai_config.DEFAULT_MODEL))
        # 同步填充 FTP / 日志配置（分享码包含全部配置）
        self.edit_ftp_host.setText(cfg.get("ftp_host", ""))
        self.edit_ftp_port.setText(str(cfg.get("ftp_port", 21)))
        self.edit_ftp_user.setText(cfg.get("ftp_user", ""))
        self.edit_ftp_pass.setText(cfg.get("ftp_pass", ""))
        self.edit_ftp_base_dir.setText(cfg.get("ftp_base_dir", ""))
        self.edit_log_url.setText(cfg.get("log_url", ""))
        self.lbl_test.setStyleSheet("color:#16a34a;")
        self.lbl_test.setText("分享码已导入（含 FTP/日志配置），点「保存」生效")

    def _export_config_file(self):
        """导出明文配置文件（供管理员分发），接收方放到程序同目录即自动生效"""
        cfg = self.get_config()
        path, _ = QFileDialog.getSaveFileName(
            self, "导出配置文件", "config.json", "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            ai_config.export_plain_config(cfg, path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"配置文件导出失败：{e}")
            return
        QMessageBox.information(
            self, "已导出",
            f"配置文件已导出到：\n{path}\n\n"
            "分发说明：接收方将该文件放到程序同目录（EXE 旁），"
            "下次启动即自动生效 AI/FTP/日志配置，无需其他操作。\n\n"
            "注意：该文件为明文（含 API Key 与 FTP 密码），请仅通过可信渠道分发。"
        )


# ---------- 主窗口 ----------

# ---------- 现代化 QSS 样式表 ----------

MODERN_STYLESHEET = """
/* ── 全局 ── */
QMainWindow {
    background: #f0f2f5;
}

QWidget#centralWidget {
    background: transparent;
}

/* ── 顶栏 ── */
QWidget#topBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #ffffff, stop:1 #f8f9fb);
    border-bottom: 1px solid #e2e6ed;
}

/* ── 标签 ── */
QLabel#fileLabel {
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    padding: 0 2px;
}

/* ── 文件路径输入框 ── */
QLineEdit#filePath {
    background: #ffffff;
    border: 1px solid #dde1e8;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    color: #334155;
    selection-background-color: #3b82f6;
}
QLineEdit#filePath:focus {
    border-color: #93b4f8;
    background: #fafbff;
}
QLineEdit#filePath:read-only {
    background: #f8f9fb;
}

/* ── 选择文件按钮 ── */
QPushButton#btnChoose {
    background: #ffffff;
    border: 1px solid #dde1e8;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #475569;
}
QPushButton#btnChoose:hover {
    background: #f0f4ff;
    border-color: #93b4f8;
    color: #2563eb;
}
QPushButton#btnChoose:pressed {
    background: #e0e7ff;
}

/* ── 设置按钮 ── */
QPushButton#btnSettings {
    background: #ffffff;
    border: 1px solid #dde1e8;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    min-width: 80px;
}
QPushButton#btnSettings:hover {
    background: #f0f4ff;
    border-color: #93b4f8;
    color: #2563eb;
}
QPushButton#btnSettings:pressed {
    background: #e0e7ff;
}

/* ── 模式切换按钮 ── */
QPushButton#btnToggleMode {
    background: #ffffff;
    border: 1px solid #dde1e8;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    min-width: 80px;
}
QPushButton#btnToggleMode:hover {
    background: #f0f4ff;
    border-color: #93b4f8;
    color: #2563eb;
}
QPushButton#btnToggleMode:pressed {
    background: #e0e7ff;
}

/* ── 零部件编号输入框 ── */
QLineEdit#inputPartNo {
    background: #ffffff;
    border: 1px solid #dde1e8;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    color: #334155;
    selection-background-color: #3b82f6;
}
QLineEdit#inputPartNo:focus {
    border-color: #93b4f8;
    background: #fafbff;
}

/* ── 开始对比按钮 ── */
QPushButton#btnCompare {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3b82f6, stop:1 #2563eb);
    color: #ffffff;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 6px 20px;
}
QPushButton#btnCompare:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2563eb, stop:1 #1d4ed8);
}
QPushButton#btnCompare:pressed {
    background: #1e40af;
}
QPushButton#btnCompare:disabled {
    background: #cbd5e1;
    color: #94a3b8;
}

/* ── WebView 容器 ── */
QWebEngineView {
    border: 1px solid #e2e6ed;
    border-radius: 8px;
}

/* ── 底部状态栏 ── */
QStatusBar {
    background: #f8f9fb;
    border-top: 1px solid #e2e6ed;
    font-size: 11px;
    color: #64748b;
    padding: 2px 8px;
}

/* ── 进度条 ── */
QProgressBar {
    background: #e2e6ed;
    border: none;
    border-radius: 4px;
    text-align: center;
    font-size: 10px;
    color: #475569;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #3b82f6, stop:1 #60a5fa);
    border-radius: 4px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self, ai_cfg):
        super().__init__()
        self.setWindowTitle("工程图纸差异对比工具")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QIcon(str(paths.resources_dir() / "app_icon.ico")))

        # AI 配置（来自用户本地设置，可在「AI 设置」对话框中修改）
        self._apply_ai_config(ai_cfg)

        # Web 目录和服务器（只读页面从包资源同步到程序目录，对比产物写入 static/）
        paths.sync_web()
        self.web_dir = paths.web_dir()
        self.static_dir = self.web_dir / "static"
        self.static_dir.mkdir(parents=True, exist_ok=True)

        self.port = find_free_port()
        self.web_server = WebServerThread(self.web_dir, self.port)
        self.web_server.start()

        self.worker = None
        self._vlm_timer = None
        self._mode = "file"  # "file" | "part_no"
        self._compare_start = 0.0
        self._init_ui()

        # 后台检查新版本（静默执行，不阻塞启动）
        self._update_thread = UpdateCheckThread()
        self._update_thread.checked.connect(self._on_update_checked)
        self._update_thread.start()

    def _apply_ai_config(self, cfg):
        """应用配置；未配置 Key 时自动降级为纯 CV 模式；按配置初始化日志上报"""
        global logbot
        self.ai_config = dict(cfg)
        self.API_KEY = cfg.get("api_key", "")
        self.AI_BASE_URL = cfg.get("base_url", ai_config.DEFAULT_BASE_URL)
        self.AI_MODEL = cfg.get("model", ai_config.DEFAULT_MODEL)
        self.AI_ENABLED = bool(cfg.get("enable_ai", True)) and bool(self.API_KEY)
        # 对比选项：线条增强与差异标注（默认开启，兼容旧配置）
        self.ENHANCE_LINES = bool(cfg.get("enhance_lines", True))
        self.ANNOTATE_CHANGES = bool(cfg.get("annotate_changes", True))
        # 日志上报（可选）：仅当配置了上报地址时启用；格式与初版一致（service_name/system 供 Loki 标签使用）
        if ai_config.log_enabled(cfg):
            logbot = LogBotSdk(url=cfg["log_url"], service_name="drawing-contrast",
                               system="xAuto", app_version=APP_VERSION)
        else:
            logbot = None

    def _init_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 0, 12, 8)
        main_layout.setSpacing(0)

        # --- 顶部栏容器 ---
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(48)
        bar_layout = QHBoxLayout(top_bar)
        bar_layout.setContentsMargins(16, 0, 16, 0)
        bar_layout.setSpacing(8)

        # 设置按钮
        self.btn_settings = QPushButton("设置")
        self.btn_settings.setObjectName("btnSettings")
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.setFixedHeight(32)
        self.btn_settings.clicked.connect(self._open_settings)
        bar_layout.addWidget(self.btn_settings)

        # 模式切换按钮
        self.btn_toggle_mode = QPushButton("文件模式")
        self.btn_toggle_mode.setObjectName("btnToggleMode")
        self.btn_toggle_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_mode.setFixedHeight(32)
        self.btn_toggle_mode.clicked.connect(self._toggle_mode)
        bar_layout.addWidget(self.btn_toggle_mode)

        # 分隔竖线
        vsep = QLabel()
        vsep.setFixedWidth(1)
        vsep.setFixedHeight(24)
        vsep.setStyleSheet("background: #dde1e8;")
        bar_layout.addWidget(vsep)

        # ── 文件模式控件 ──
        # 旧图
        self.lbl_b = QLabel("旧图")
        self.lbl_b.setObjectName("fileLabel")
        bar_layout.addWidget(self.lbl_b)
        self.path_b = QLineEdit()
        self.path_b.setObjectName("filePath")
        self.path_b.setReadOnly(True)
        self.path_b.setPlaceholderText("选择变更前 PDF …")
        bar_layout.addWidget(self.path_b, stretch=1)
        self.btn_b = QPushButton("浏览")
        self.btn_b.setObjectName("btnChoose")
        self.btn_b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_b.clicked.connect(lambda: self._choose_pdf('b'))
        bar_layout.addWidget(self.btn_b)

        # 分隔符
        self.sep = QLabel("→")
        self.sep.setStyleSheet("color: #94a3b8; font-size: 16px; padding: 0 4px;")
        bar_layout.addWidget(self.sep)

        # 新图
        self.lbl_a = QLabel("新图")
        self.lbl_a.setObjectName("fileLabel")
        bar_layout.addWidget(self.lbl_a)
        self.path_a = QLineEdit()
        self.path_a.setObjectName("filePath")
        self.path_a.setReadOnly(True)
        self.path_a.setPlaceholderText("选择变更后 PDF …")
        bar_layout.addWidget(self.path_a, stretch=1)
        self.btn_a = QPushButton("浏览")
        self.btn_a.setObjectName("btnChoose")
        self.btn_a.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_a.clicked.connect(lambda: self._choose_pdf('a'))
        bar_layout.addWidget(self.btn_a)

        # ── 零部件编号模式控件（初始隐藏，需在「设置」中配置 FTP 后启用） ──
        self.lbl_part_b = QLabel("旧图")
        self.lbl_part_b.setObjectName("fileLabel")
        self.input_part_b = QLineEdit()
        self.input_part_b.setObjectName("inputPartNo")
        self.input_part_b.setPlaceholderText("输入旧图零部件编号 …")
        self.lbl_part_a = QLabel("新图")
        self.lbl_part_a.setObjectName("fileLabel")
        self.input_part_a = QLineEdit()
        self.input_part_a.setObjectName("inputPartNo")
        self.input_part_a.setPlaceholderText("输入新图零部件编号 …")
        self.sep_part = QLabel("→")
        self.sep_part.setStyleSheet("color: #94a3b8; font-size: 16px; padding: 0 4px;")

        # 零部件编号控件加入布局并隐藏
        for w in [self.lbl_part_b, self.input_part_b, self.sep_part, self.lbl_part_a, self.input_part_a]:
            bar_layout.addWidget(w)
            w.hide()

        # 对比按钮
        bar_layout.addSpacing(8)
        self.btn_compare = QPushButton("开始对比")
        self.btn_compare.setObjectName("btnCompare")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.setFixedHeight(32)
        self.btn_compare.clicked.connect(self._start_comparison)
        bar_layout.addWidget(self.btn_compare)

        main_layout.addWidget(top_bar)
        main_layout.addSpacing(6)

        # --- 内嵌浏览器（占满剩余空间） ---
        self.web_view = QWebEngineView()
        self.web_view.setHtml(self._placeholder_html())
        main_layout.addWidget(self.web_view, stretch=1)

        # --- 底部状态栏 ---
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        self.status_label = QLabel("就绪")
        status_bar.addWidget(self.status_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(14)
        status_bar.addPermanentWidget(self.progress_bar)

    def _placeholder_html(self):
        return """
        <html>
        <body style="display:flex;align-items:center;justify-content:center;height:100vh;
                     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,'Microsoft YaHei',sans-serif;
                     color:#94a3b8;background:#f0f2f5;margin:0;">
            <div style="text-align:center;">
                <div style="font-size:48px;margin-bottom:16px;opacity:0.6;">📐</div>
                <p style="font-size:15px;font-weight:600;color:#64748b;margin-bottom:6px;">工程图纸差异对比</p>
                <p style="font-size:13px;">选择旧图与新图后点击「开始对比」</p>
            </div>
        </body>
        </html>
        """

    def _open_settings(self):
        """打开设置对话框，保存后立即应用到当前会话"""
        dlg = SettingsDialog(self.ai_config, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cfg = dlg.get_config()
            try:
                ai_config.save_config(cfg)
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"配置保存失败：{e}")
                return
            self._apply_ai_config(cfg)
            ai_state = "已启用" if self.AI_ENABLED else "不可用（未配置 API Key）"
            ftp_state = "已启用" if ai_config.ftp_enabled(cfg) else "未配置"
            log_state = "已启用" if ai_config.log_enabled(cfg) else "未配置"
            self.status_label.setText(
                f"设置已保存 — AI：{ai_state}，FTP：{ftp_state}，日志：{log_state}"
            )
            log.info(
                f"设置已更新: model={self.AI_MODEL}, "
                f"ftp_enabled={ai_config.ftp_enabled(cfg)}, log_enabled={ai_config.log_enabled(cfg)}"
            )

    def _choose_pdf(self, which):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 文件", "", "PDF 文件 (*.pdf)"
        )
        if path:
            if which == 'a':
                self.path_a.setText(path)
            else:
                self.path_b.setText(path)

    def _toggle_mode(self):
        """切换 文件模式 / 零部件编号模式"""
        file_widgets = [self.lbl_b, self.path_b, self.btn_b,
                        self.sep, self.lbl_a, self.path_a, self.btn_a]
        part_widgets = [self.lbl_part_b, self.input_part_b, self.sep_part,
                        self.lbl_part_a, self.input_part_a]
        if self._mode == "file":
            self._mode = "part_no"
            self.btn_toggle_mode.setText("零部件编号模式")
            for w in file_widgets:
                w.hide()
            for w in part_widgets:
                w.show()
        else:
            self._mode = "file"
            self.btn_toggle_mode.setText("文件模式")
            for w in part_widgets:
                w.hide()
            for w in file_widgets:
                w.show()

    def _start_comparison(self):
        """启动对比，按当前模式分发"""
        if self._mode == "part_no":
            self._start_comparison_part_mode()
            return

        pdf_a = self.path_a.text().strip()
        pdf_b = self.path_b.text().strip()

        if not pdf_a or not pdf_b:
            QMessageBox.warning(self, "提示", "请先选择旧图和新图的 PDF 文件")
            return

        if not os.path.exists(pdf_a):
            QMessageBox.warning(self, "错误", f"文件不存在: {pdf_a}")
            return
        if not os.path.exists(pdf_b):
            QMessageBox.warning(self, "错误", f"文件不存在: {pdf_b}")
            return

        self._launch_comparison(pdf_a, pdf_b)

    def _start_comparison_part_mode(self):
        """零部件编号模式：从 FTP 下载两份 PDF 后执行对比（FTP 未配置时引导到设置）"""
        if not ai_config.ftp_enabled(self.ai_config):
            QMessageBox.information(
                self, "FTP 未配置",
                "零部件编号模式需要 FTP 服务器支持。\n请在左上角「设置」→「FTP 设置」中填写服务器信息。"
            )
            return

        part_b = self.input_part_b.text().strip()
        part_a = self.input_part_a.text().strip()
        if not part_b or not part_a:
            QMessageBox.warning(self, "提示", "请输入旧图和新图的零部件编号")
            return

        try:
            downloader = FtpDownloader(
                host=self.ai_config.get("ftp_host", ""),
                port=self.ai_config.get("ftp_port", 21),
                user=self.ai_config.get("ftp_user", ""),
                password=self.ai_config.get("ftp_pass", ""),
                base_dir=self.ai_config.get("ftp_base_dir", ""),
            )
        except FtpDownloadError as e:
            QMessageBox.critical(self, "FTP 错误", str(e))
            return

        tmp_dir = os.path.join(str(self.static_dir), "ftp_cache")
        try:
            self.status_label.setText(f"正在下载零部件编号 {part_b} ...")
            QApplication.processEvents()
            pdf_b = downloader.download_by_part_no(part_b, tmp_dir)
            self.status_label.setText(f"正在下载零部件编号 {part_a} ...")
            QApplication.processEvents()
            pdf_a = downloader.download_by_part_no(part_a, tmp_dir)
        except FtpDownloadError as e:
            if logbot:
                logbot.error("FTP 下载失败", "MainWindow", "ftp_download")
            self.status_label.setText("FTP 下载失败")
            QMessageBox.critical(self, "FTP 下载失败", str(e))
            return

        self._launch_comparison(pdf_a, pdf_b)

    def _launch_comparison(self, pdf_a: str, pdf_b: str):
        """启动对比工作线程"""
        self.btn_compare.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在处理...")
        self._compare_start = time.time()
        if logbot:
            logbot.info("开始对比", "MainWindow", "launch",
                        extra={"mode": self._mode, "ai_enabled": self.AI_ENABLED})
        # 停止上一轮的 VLM 状态轮询，避免误读新一轮的 meta.json
        if self._vlm_timer:
            self._vlm_timer.stop()

        self.worker = ComparisonWorker(
            pdf_a, pdf_b, str(self.static_dir),
            self.API_KEY, self.AI_ENABLED, self.AI_MODEL,
            base_url=self.AI_BASE_URL,
            enhance_lines=self.ENHANCE_LINES,
            annotate_changes=self.ANNOTATE_CHANGES
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, meta):
        self.progress_bar.setValue(100)
        self._update_result_status(meta)
        self.btn_compare.setEnabled(True)
        if logbot:
            elapsed = int((time.time() - self._compare_start) * 1000)
            logbot.info("对比完成", "MainWindow", "finished", elapsed_ms=elapsed)

        # 加载对比页面
        url = f"http://127.0.0.1:{self.port}/index.html"
        self.web_view.setUrl(QUrl(url))

        # VLM 后台解读期间轮询 meta.json，完成后刷新状态栏统计
        if meta.get('vlm_status') == 'running':
            if self._vlm_timer is None:
                self._vlm_timer = QTimer(self)
                self._vlm_timer.setInterval(2000)
                self._vlm_timer.timeout.connect(self._poll_vlm_status)
            self._vlm_timer.start()

    def _poll_vlm_status(self):
        """轮询 meta.json，VLM 后台解读结束后更新状态栏"""
        try:
            with open(self.static_dir / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return
        if meta.get('vlm_status') == 'running':
            regions = meta.get('regions') or []
            done = sum(1 for r in regions if 'change_type' in r)
            self.status_label.setText(f"对比完成 — VLM解读差异区域中 ({done}/{len(regions)})...")
            return
        self._vlm_timer.stop()
        self._update_result_status(meta)

    def _update_result_status(self, meta):
        """根据 meta 刷新状态栏结果文案"""
        diff_note = " ⚠️ [不同零件图纸]" if meta.get('different_drawings') else ""
        # 区域级变更统计（新版 meta 才有 regions/moved_views）
        regions = meta.get('regions') or []
        moved_views = meta.get('moved_views') or []
        real_changes = [r for r in regions if r.get('is_real_change') is not False]
        if regions or moved_views:
            region_note = f" — 检测到 {len(real_changes) + len(moved_views)} 处变更"
            if moved_views:
                region_note += f"（其中 {len(moved_views)} 处视图移动）"
        else:
            region_note = ""
        vlm_note = " — VLM后台解读中..." if meta.get('vlm_status') == 'running' else ""
        self.status_label.setText(
            f"对比完成{diff_note}{region_note} — 新增: {meta['added_pixels']} 像素, "
            f"删除: {meta['removed_pixels']} 像素{vlm_note}"
        )

    def _on_error(self, err_msg):
        self.progress_bar.setValue(0)
        self.status_label.setText(f"错误: {err_msg}")
        self.btn_compare.setEnabled(True)
        if logbot:
            elapsed = int((time.time() - self._compare_start) * 1000)
            logbot.error("对比失败", "MainWindow", "error", elapsed_ms=elapsed)
        QMessageBox.critical(self, "对比失败", err_msg)

    def _on_update_checked(self, release):
        """更新检查结果回调：发现新版本时弹窗提醒，其余情况仅记日志"""
        if not release:
            log.info("更新检查：未获取到 Release 信息（网络不可用或尚无 Release）")
            return
        if release["version"] <= parse_version(APP_VERSION):
            log.info(f"更新检查：当前版本 {APP_VERSION} 已是最新（远端 {release['tag']}）")
            return
        log.info(f"更新检查：发现新版本 {release['tag']}（当前 {APP_VERSION}）")
        box = QMessageBox(self)
        box.setWindowTitle("发现新版本")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"检测到新版本 {release['tag']}（当前版本 {APP_VERSION}）")
        box.setInformativeText("请前往 GitHub Release 页面下载新版 DrawingContrast.exe 覆盖更新。")
        btn_go = box.addButton("前往下载", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("暂不更新", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() == btn_go:
            url = release["download_url"] or release["html_url"]
            if url:
                QDesktopServices.openUrl(QUrl(url))

    def closeEvent(self, event):
        self.web_server.stop()
        super().closeEvent(event)


# ---------- Splash Screen ----------

def create_splash_pixmap():
    """生成启动画面 Pixmap"""
    w, h = 420, 260
    pixmap = QPixmap(w, h)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 背景渐变
    grad = QLinearGradient(0, 0, w, h)
    grad.setColorAt(0, QColor(30, 58, 138))    # 深蓝
    grad.setColorAt(1, QColor(37, 99, 235))    # 亮蓝
    painter.setBrush(grad)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, w, h, 16, 16)

    # 应用名称
    painter.setPen(QColor(255, 255, 255))
    font_title = QFont("Microsoft YaHei", 18, QFont.Weight.Bold)
    painter.setFont(font_title)
    painter.drawText(0, 60, w, 40, Qt.AlignmentFlag.AlignCenter, "工程图纸差异对比工具")

    # 副标题
    font_sub = QFont("Microsoft YaHei", 10)
    painter.setFont(font_sub)
    painter.setPen(QColor(200, 220, 255))
    painter.drawText(0, 105, w, 30, Qt.AlignmentFlag.AlignCenter, "Drawing Contrast Tool")

    # 加载提示
    font_status = QFont("Microsoft YaHei", 9)
    painter.setFont(font_status)
    painter.setPen(QColor(180, 200, 240))
    painter.drawText(0, 158, w, 25, Qt.AlignmentFlag.AlignCenter, "正在初始化...")

    # 版权（预留底部空间给 showMessage，避免文字堆叠）
    font_copy = QFont("Microsoft YaHei", 8)
    painter.setFont(font_copy)
    painter.setPen(QColor(140, 160, 200))
    painter.drawText(0, 190, w, 16, Qt.AlignmentFlag.AlignCenter, "© 2026 xcode.im · Xing")
    painter.drawText(0, 208, w, 16, Qt.AlignmentFlag.AlignCenter, "github.com/ayumilove/DrawingContrast")

    painter.end()
    return pixmap


# ---------- 入口 ----------

def main():
    global log
    log = setup_logger()

    # 为每个实例创建独立的 WebEngine 数据目录，避免多开冲突
    tmp_profile_dir = tempfile.mkdtemp(prefix="DrawingContrast_")
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"--user-data-dir={tmp_profile_dir}"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 立即显示 Splash Screen
    splash_pixmap = create_splash_pixmap()
    splash = QSplashScreen(splash_pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    splash.showMessage(
        "正在加载本地配置...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(180, 200, 240)
    )
    app.processEvents()

    # 隔离 WebEngine profile 存储路径
    profile = QWebEngineProfile.defaultProfile()
    profile.setCachePath(os.path.join(tmp_profile_dir, "cache"))
    profile.setPersistentStoragePath(os.path.join(tmp_profile_dir, "data"))

    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 应用现代化样式表
    app.setStyleSheet(MODERN_STYLESHEET)

    # 加载用户本地设置（BYOK：用户自带 Key，通过应用内「设置」对话框维护，支持配置文件分发）
    cfg = ai_config.load_config()
    if not cfg.get("api_key"):
        log.info("未配置 API Key，AI 功能不可用，基础对比功能正常。可通过界面「设置」配置")

    splash.showMessage(
        "正在加载界面...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(180, 200, 240)
    )
    app.processEvents()

    window = MainWindow(cfg)
    window.show()

    # 日志上报（可选）：仅当配置了上报地址时上报启动事件（不含本地主机信息）
    if logbot:
        logbot.info("程序启动", "App", "main", extra={"ai_enabled": window.AI_ENABLED})

    # 主窗口显示后关闭 splash
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


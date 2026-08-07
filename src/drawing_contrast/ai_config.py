"""
AI 配置模块：用户自带 Key（BYOK），代码与仓库中不含任何密钥。

配置文件：与程序同目录的 config.json（打包后在 EXE 旁，源码运行时在项目根目录）。
Key 使用 Windows DPAPI（CryptProtectData，AES 加密 + 当前用户凭据）加密保存，
并叠加程序专属熵值：仅本机同一 Windows 用户可解密，配置文件被拷到其他机器/用户无法还原。

Key 生效优先级（高 → 低）：
  1. 环境变量 SILICONFLOW_API_KEY / AI_API_KEY（Key）与 AI_BASE_URL / AI_MODEL
  2. 本地加密配置文件
  3. 内置默认值（Key 为空，AI 功能关闭，纯 CV 对比仍可用）
"""

import os
import sys
import json
import base64
from pathlib import Path

from drawing_contrast import paths

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# DPAPI 附加熵：只有本程序能解密，其他程序即使同用户也解不开
_ENTROPY = b"DrawingContrast-v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1

_IS_WINDOWS = sys.platform == "win32"

# 旧版配置位置（用户目录），仅用于读取兼容
_LEGACY_CONFIG_PATH = Path.home() / ".drawing_contrast" / "config.json"


def app_dir() -> Path:
    """程序所在目录：PyInstaller 打包后为 EXE 旁，源码运行时为项目根目录"""
    return paths.program_dir()


def config_path() -> Path:
    return app_dir() / "config.json"


def _default_config() -> dict:
    return {
        "api_key": "",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "enable_ai": True,
        # FTP 零部件编号下载（可选）：host 为空则功能不启用
        "ftp_host": "",
        "ftp_port": 21,
        "ftp_user": "",
        "ftp_pass": "",
        "ftp_base_dir": "",
        # 日志上报（可选）：url 为空则不上报
        "log_url": "",
    }


def ftp_enabled(cfg: dict) -> bool:
    """FTP 功能启用判定：主机与用户名均已配置"""
    return bool(cfg.get("ftp_host")) and bool(cfg.get("ftp_user"))


def log_enabled(cfg: dict) -> bool:
    """日志上报启用判定：已配置上报地址"""
    return bool(cfg.get("log_url"))


# ---------- DPAPI 加解密（Windows） ----------

def _dpapi_call(func, data: bytes) -> bytes:
    """调用 crypt32 的 CryptProtectData / CryptUnprotectData"""
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    def make_blob(buf: bytes) -> DATA_BLOB:
        return DATA_BLOB(len(buf), ctypes.create_string_buffer(buf, len(buf)))

    blob_in = make_blob(data)
    blob_entropy = make_blob(_ENTROPY)
    blob_out = DATA_BLOB()
    if not func(ctypes.byref(blob_in), None, ctypes.byref(blob_entropy),
                None, None, _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out)):
        raise OSError(f"DPAPI 调用失败 (WinError {ctypes.GetLastError()})")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _protect(data: bytes) -> bytes:
    import ctypes
    return _dpapi_call(ctypes.windll.crypt32.CryptProtectData, data)


def _unprotect(data: bytes) -> bytes:
    import ctypes
    return _dpapi_call(ctypes.windll.crypt32.CryptUnprotectData, data)


# ---------- 配置读写 ----------

def _read_file(path: Path) -> dict:
    """读取配置文件（自动识别加密/明文格式），失败返回空 dict"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        if raw.get("encrypted") and raw.get("data"):
            if not _IS_WINDOWS:
                print(f"警告: 加密配置仅支持在 Windows 上解密，忽略 {path}")
                return {}
            payload = _unprotect(base64.b64decode(raw["data"]))
            saved = json.loads(payload.decode("utf-8"))
        else:
            saved = raw  # 兼容旧版明文配置
        return saved if isinstance(saved, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        # 解密失败（文件被拷到其他机器/用户）或内容损坏：视为未配置，不落任何明文
        print(f"警告: 读取 AI 配置失败（{e}），将视为未配置")
        return {}


def load_config() -> dict:
    """读取本地配置，并用环境变量覆盖（便于 CI / 脚本场景）"""
    cfg = _default_config()

    path = config_path()
    saved = _read_file(path)
    if not saved and not path.exists():
        # 兼容旧版：用户目录下的配置仍可读（只读，下次保存会写到程序目录）
        saved = _read_file(_LEGACY_CONFIG_PATH)
    cfg.update({k: v for k, v in saved.items() if k in cfg})

    # 环境变量覆盖
    env_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("AI_API_KEY")
    if env_key:
        cfg["api_key"] = env_key.strip()
    env_base = os.environ.get("AI_BASE_URL")
    if env_base:
        cfg["base_url"] = env_base.strip()
    env_model = os.environ.get("AI_MODEL")
    if env_model:
        cfg["model"] = env_model.strip()

    cfg["base_url"] = normalize_base_url(cfg["base_url"] or DEFAULT_BASE_URL)
    return cfg


def save_config(cfg: dict) -> Path:
    """保存配置到程序目录；Windows 上对内容整体 DPAPI 加密，失败抛出异常"""
    data = _default_config()
    data.update({k: v for k, v in cfg.items() if k in data})
    data["base_url"] = normalize_base_url(data["base_url"] or DEFAULT_BASE_URL)

    path = config_path()
    if _IS_WINDOWS:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        protected = _protect(payload)
        content = json.dumps({
            "v": 1,
            "encrypted": True,
            "data": base64.b64encode(protected).decode("ascii"),
        }, indent=2)
    else:
        # 非 Windows（仅开发场景）：明文保存
        content = json.dumps(data, indent=2, ensure_ascii=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def export_plain_config(cfg: dict, path) -> Path:
    """导出明文配置文件（用于管理员分发给其他用户）。
    接收方将文件放到程序同目录即可自动生效（load_config 兼容明文格式）。
    注意：明文含 API Key，仅应通过可信渠道分发。
    """
    data = _default_config()
    data.update({k: v for k, v in cfg.items() if k in data})
    data["base_url"] = normalize_base_url(data["base_url"] or DEFAULT_BASE_URL)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def normalize_base_url(url: str) -> str:
    """规范化 OpenAI 兼容 Base URL：去尾部斜杠，统一形如 https://api.siliconflow.cn/v1"""
    return (url or "").strip().rstrip("/")


# ---------- 分享码（跨机器分发配置） ----------

_TOKEN_PREFIX = "DC1:"


def encode_share_token(cfg: dict) -> str:
    """把配置打包成可跨机器传递的分享码（含明文 Key，仅应在可信渠道传递）"""
    data = _default_config()
    data.update({k: v for k, v in cfg.items() if k in data})
    data["base_url"] = normalize_base_url(data["base_url"] or DEFAULT_BASE_URL)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _TOKEN_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def decode_share_token(token: str) -> dict:
    """解析分享码，失败抛 ValueError"""
    s = (token or "").strip()
    if not s.startswith(_TOKEN_PREFIX):
        raise ValueError("不是有效的分享码（缺少 DC1: 前缀）")
    try:
        payload = base64.urlsafe_b64decode(s[len(_TOKEN_PREFIX):])
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        raise ValueError("分享码内容损坏，无法解析")
    if not isinstance(data, dict) or "api_key" not in data:
        raise ValueError("分享码格式不正确")
    data["base_url"] = normalize_base_url(data.get("base_url") or DEFAULT_BASE_URL)
    return {k: v for k, v in data.items() if k in _default_config()}


def chat_completions_url(base_url: str) -> str:
    """由 Base URL 拼出 chat/completions 端点"""
    base = normalize_base_url(base_url or DEFAULT_BASE_URL)
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"

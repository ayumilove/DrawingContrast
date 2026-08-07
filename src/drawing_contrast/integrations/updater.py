"""
GitHub Release 更新检查（可选功能）

- 启动时后台请求 GitHub Releases API，与当前版本比较，发现新版本由 UI 层提醒用户
- 任何异常（断网、无 Release、限流等）均静默降级，绝不影响主业务流程
- 公开仓库匿名访问 API 即可，无需 token
"""

import requests

# GitHub Releases API（公开仓库，匿名访问）
RELEASE_API = "https://api.github.com/repos/ayumilove/DrawingContrast/releases/latest"


def parse_version(s: str) -> tuple:
    """将 'v0.3.0' / '0.3.0' 解析为 int 元组，无法解析时返回 (0,)"""
    s = (s or "").strip().lstrip("vV")
    parts = []
    for seg in s.split("."):
        digits = "".join(ch for ch in seg if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def fetch_latest_release(timeout: int = 5):
    """
    获取最新 Release 信息。

    返回 dict：{tag, version, html_url, download_url}；
    请求失败、无 Release 或被限流时返回 None。
    """
    try:
        resp = requests.get(
            RELEASE_API, timeout=timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        tag = data.get("tag_name", "")
        # 优先取首个 .exe 资产的直链，无资产时回退到 Release 页面
        download_url = None
        for asset in data.get("assets", []) or []:
            name = asset.get("name", "")
            if name.lower().endswith(".exe"):
                download_url = asset.get("browser_download_url")
                break
        return {
            "tag": tag,
            "version": parse_version(tag),
            "html_url": data.get("html_url", ""),
            "download_url": download_url,
        }
    except Exception:
        return None

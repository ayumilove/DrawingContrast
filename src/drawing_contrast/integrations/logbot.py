"""
LogBot 日志上报 SDK（可选功能，配置驱动）

- 上报地址来自用户配置（ai_config 的 log_url），未配置则完全不启用
- 上报失败静默降级，绝不影响主业务流程
- 隐私原则：不采集主机名/IP/MAC 等本地信息，不上报图纸内容与 API Key，
  仅上报业务级事件（启动/对比成功失败等）

上报格式与初版保持一致（Loki/Grafana 依赖）：
  service_name=drawing-contrast、system 标签、timestamp 为 epoch 秒级整数
服务端要求（见 README「可选功能」章节）：
  接收 JSON POST 请求，字段见 _post() 中的 payload
"""

import time
import requests


class LogBotSdk:
    def __init__(self, url: str, service_name: str = "drawing-contrast",
                 system: str = "", app_version: str = "", timeout: int = 5):
        self.url = url
        self.service_name = service_name
        self.system = system
        self.app_version = app_version
        self.timeout = timeout

    def _post(self, level: str, message: str, class_name: str = "",
              method_name: str = "", elapsed_ms: int = None, extra: dict = None):
        payload = {
            "timestamp": int(time.time()),
            "level": level,
            "message": message,
            "service_name": self.service_name,
        }
        if self.system:
            payload["system"] = self.system
        if self.app_version:
            payload["app_version"] = self.app_version
        if class_name:
            payload["class_name"] = class_name
        if method_name:
            payload["method_name"] = method_name
        if elapsed_ms is not None:
            payload["elapsed_ms"] = elapsed_ms
        if extra:
            payload["extra"] = extra
        try:
            requests.post(
                self.url, json=payload, timeout=self.timeout,
                proxies={"http": None, "https": None}  # 直连，绕过系统代理
            )
        except Exception:
            pass  # 上报失败静默降级，不影响主流程

    def info(self, message: str, class_name: str = "", method_name: str = "",
             elapsed_ms: int = None, extra: dict = None):
        self._post("info", message, class_name, method_name, elapsed_ms, extra)

    def warning(self, message: str, class_name: str = "", method_name: str = "",
                elapsed_ms: int = None, extra: dict = None):
        self._post("warning", message, class_name, method_name, elapsed_ms, extra)

    def error(self, message: str, class_name: str = "", method_name: str = "",
              elapsed_ms: int = None, extra: dict = None):
        self._post("error", message, class_name, method_name, elapsed_ms, extra)

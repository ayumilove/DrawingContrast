"""
简单的Web服务器，用于展示图纸差异可视化页面
用法：uv run scripts/web_server.py
"""
import http.server

from drawing_contrast import paths


class DiffViewerHandler(http.server.SimpleHTTPRequestHandler):
    """自定义Handler，将 /static 指向 web/static 目录"""

    def __init__(self, *args, **kwargs):
        self.web_dir = str(paths.web_dir())
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()


def main():
    port = 8765
    server = http.server.HTTPServer(("0.0.0.0", port), DiffViewerHandler)
    print(f"差异可视化服务已启动: http://localhost:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()

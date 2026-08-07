"""
FTP 零部件编号下载器（可选功能，配置驱动）

- 服务器参数全部来自用户配置（ai_config），代码中不含任何硬编码凭证
- 未配置 FTP 时该功能不启用（界面给出引导提示）
- 兼容禁用 NLST/LIST 的服务器：不做目录列举，直接按候选路径 RETR 下载

候选路径策略（按序尝试，命中即返回）：
  1. <基础目录>/<前缀>/<零部件编号>.pdf   （前缀 = 零部件编号首段，如 8DM-2-1350-275-00-WH → 8DM）
  2. <前缀>/<零部件编号>.pdf
  3. <基础目录>/<零部件编号>/<零部件编号>.pdf （零部件编号同名子目录）
  4. <零部件编号>/<零部件编号>.pdf
  5. <基础目录>/<零部件编号>.pdf
  6. <零部件编号>.pdf                        （根目录平铺）
"""

import ftplib
from pathlib import Path


class FtpDownloadError(Exception):
    """FTP 下载失败（连接/登录/文件不存在等），携带可直接展示给用户的信息"""
    pass


class FtpDownloader:
    def __init__(self, host, port=21, user="", password="", base_dir="", timeout=15):
        if not host or not user:
            raise FtpDownloadError("FTP 未配置：请先在「设置 → FTP 设置」中填写服务器信息")
        self.host = host
        self.port = int(port or 21)
        self.user = user
        self.password = password or ""
        self.base_dir = (base_dir or "").strip().strip("/")
        self.timeout = timeout

    def _candidate_paths(self, part_no: str):
        """生成候选远程路径列表（不做目录列举，直接尝试 RETR）"""
        fname = f"{part_no}.pdf"
        prefix = part_no.split("-")[0] if "-" in part_no else ""
        paths = []
        # 前缀子目录组织（如 8DM/8DM-2-1350-275-00-WH.pdf）
        if prefix:
            if self.base_dir:
                paths.append(f"{self.base_dir}/{prefix}/{fname}")
            paths.append(f"{prefix}/{fname}")
        # 零部件编号同名子目录组织
        if self.base_dir:
            paths.append(f"{self.base_dir}/{part_no}/{fname}")
        paths.append(f"{part_no}/{fname}")
        # 平铺组织
        if self.base_dir:
            paths.append(f"{self.base_dir}/{fname}")
        paths.append(fname)
        return paths

    def download_by_part_no(self, part_no: str, dest_dir: str) -> str:
        """按零部件编号下载 PDF 到 dest_dir，返回本地文件路径"""
        part_no = part_no.strip()
        if not part_no:
            raise FtpDownloadError("零部件编号不能为空")

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        local_path = dest / f"{part_no}.pdf"

        ftp = ftplib.FTP()
        try:
            ftp.connect(host=self.host, port=self.port, timeout=self.timeout)
        except Exception as e:
            raise FtpDownloadError(
                f"无法连接 FTP 服务器 {self.host}:{self.port}\n错误: {e}\n请检查网络与「设置」中的 FTP 配置"
            )

        try:
            ftp.login(user=self.user, passwd=self.password)
        except ftplib.error_perm as e:
            raise FtpDownloadError(f"FTP 登录失败（用户名或密码错误）: {e}")
        except Exception as e:
            raise FtpDownloadError(f"FTP 登录异常: {e}")

        try:
            ftp.set_pasv(True)
            last_err = None
            for remote_path in self._candidate_paths(part_no):
                try:
                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {remote_path}", f.write)
                    # 下载成功但文件为空视为不存在
                    if local_path.stat().st_size == 0:
                        local_path.unlink(missing_ok=True)
                        last_err = f"{remote_path}: 文件为空"
                        continue
                    return str(local_path)
                except ftplib.error_perm as e:
                    # 550 等：该候选路径不存在，尝试下一个
                    local_path.unlink(missing_ok=True)
                    last_err = str(e)
                except ftplib.error_temp as e:
                    local_path.unlink(missing_ok=True)
                    last_err = str(e)
            raise FtpDownloadError(
                f"在 FTP 服务器上未找到零部件编号 {part_no} 对应的图纸\n"
                f"已尝试路径: {', '.join(self._candidate_paths(part_no))}\n"
                f"最后错误: {last_err}"
            )
        finally:
            try:
                ftp.quit()
            except Exception:
                pass

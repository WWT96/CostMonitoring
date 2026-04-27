"""配置管理
========
读取顺序：项目根目录 .env 文件 → 系统环境变量 → 内置默认值。
部署时在同目录下创建 .env 文件，或直接设置系统环境变量。
"""
from __future__ import annotations

import os
from pathlib import Path

# 加载 .env 文件（优先于系统环境变量；文件不存在时静默跳过）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv 未安装时回退到系统环境变量


class _Settings:
    """通过属性动态读取环境变量，确保每次调用都能获取最新值。"""

    @property
    def api_auth_token(self) -> str:
        """API Bearer Token，企业系统调用时需在请求头中携带。
        生产环境务必替换为高强度随机字符串（建议 32 位以上）。"""
        return os.getenv("API_AUTH_TOKEN", "changeme-please-rotate")

    @property
    def enterprise_system_url(self) -> str:
        """企业 Java 系统的 API 地址（保留字段，供未来主动拉取数据使用）。"""
        return os.getenv("ENTERPRISE_SYSTEM_URL", "")

    @property
    def api_data_cache_path(self) -> str:
        """API 服务持久化缓存的 Parquet 文件路径（相对于项目根目录）。"""
        return os.getenv("API_DATA_CACHE_PATH", "api_cache.parquet")

    @property
    def api_host(self) -> str:
        """API 服务监听地址。"""
        return os.getenv("API_HOST", "0.0.0.0")

    @property
    def api_port(self) -> int:
        """API 服务监听端口。"""
        return int(os.getenv("API_PORT", "8000"))

    @property
    def db_url(self) -> str:
        """Supabase PostgreSQL 连接串。"""
        return os.getenv("DB_URL", "").strip()

    @property
    def reset_cost_anomaly_results_on_start(self) -> bool:
        """是否在启动初始化时强制重置 cost_anomaly_results 表。"""
        return os.getenv("RESET_COST_ANOMALY_RESULTS_ON_START", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


settings = _Settings()

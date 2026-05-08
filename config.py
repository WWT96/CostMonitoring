"""配置管理
========
读取顺序：初始化参数 → 系统环境变量 → 项目根目录 .env 文件 → 内置默认值。
部署时在同目录下创建 .env 文件，或直接设置系统环境变量。
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic.v1 import BaseSettings, Field, validator
except ImportError:
    from pydantic import BaseSettings, Field, validator  # type: ignore


_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """统一配置入口，使用 Field(..., env=...) 显式映射环境变量。"""

    api_auth_token: str = Field("changeme-please-rotate", env="API_AUTH_TOKEN")
    enterprise_system_url: str = Field("", env="ENTERPRISE_SYSTEM_URL")
    api_data_cache_path: str = Field("api_cache.parquet", env="API_DATA_CACHE_PATH")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    db_url: str = Field("", env="DB_URL")
    llm_api_key: str = Field("", env="LLM_API_KEY")
    llm_api_base_url: str = Field("https://api.deepseek.com", env="LLM_API_BASE_URL")
    llm_api_model: str = Field("deepseek-chat", env="LLM_API_MODEL")
    llm_timeout_seconds: int = Field(45, env="LLM_TIMEOUT_SECONDS")
    llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
    reset_cost_anomaly_results_on_start: bool = Field(False, env="RESET_COST_ANOMALY_RESULTS_ON_START")

    @validator(
        "api_auth_token",
        "enterprise_system_url",
        "api_data_cache_path",
        "api_host",
        "db_url",
        "llm_api_key",
        "llm_api_base_url",
        "llm_api_model",
        pre=True,
    )
    def _strip_string_values(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def llm_model(self) -> str:
        """兼容旧字段名；统一转发到 llm_api_model。"""
        return self.llm_api_model

    @property
    def llm_enabled(self) -> bool:
        """LLM 是否已配置可用的 API Key。"""
        return bool(self.llm_api_key)


settings = Settings()

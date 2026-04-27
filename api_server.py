"""企业系统集成 API 服务层
======================
运行方式：
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

认证：
  所有数据接口需要在 Authorization 请求头中携带 Bearer Token。
  Token 值由 .env 文件中的 API_AUTH_TOKEN 配置。
  示例：  Authorization: Bearer your-token-here

交互文档（开发模式下访问）：
  http://localhost:8000/docs   — Swagger UI
  http://localhost:8000/redoc  — ReDoc
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

import processor
from config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="备件成本监控 API",
    description=(
        "企业系统集成接口层，支持通过 REST API 推送数据并获取异常分析结果。\n\n"
        "**所有数据接口需要在 `Authorization` 请求头中携带 Bearer Token。**"
    ),
    version="1.0.0",
)

_security = HTTPBearer()

# 线程池：用于将 CPU-bound 的 pandas / sklearn 计算移出异步事件循环
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# 进程内缓存（同时持久化至 Supabase PostgreSQL，供 Streamlit app.py 侧读取）
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "df": None,          # pd.DataFrame | None
    "price_col": None,   # str | None
    "updated_at": None,  # datetime | None
}


# ---------------------------------------------------------------------------
# 鉴权依赖
# ---------------------------------------------------------------------------
def _verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> None:
    """验证 Bearer Token，不匹配时返回 401。"""
    if credentials.credentials != settings.api_auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Token，请检查 Authorization: Bearer <token> 请求头。",
        )


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------
def _persist_cache(df: pd.DataFrame, price_col: Optional[str]) -> None:
    """将 DataFrame 写入 Supabase 的 core_cost_records 表。"""
    try:
        processor.persist_core_cost_records(df, price_col=price_col, mode="full")
    except Exception as exc:
        logger.warning("数据库持久化失败（数据仍保留在内存中）: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic 请求模型
# ---------------------------------------------------------------------------
class SyncRequest(BaseModel):
    records: List[Dict[str, Any]]
    mode: Literal["full", "incremental"] = "incremental"


class AnomalyRequest(BaseModel):
    records: List[Dict[str, Any]]
    price_col: Optional[str] = None


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/health", summary="健康检查（无需认证）", tags=["监控"])
def health() -> dict:
    """返回服务状态及当前内存缓存概况，无需认证。"""
    df: Optional[pd.DataFrame] = _state["df"]
    return {
        "status": "ok",
        "cached_records": int(len(df)) if df is not None else 0,
        "cache_updated_at": (
            _state["updated_at"].isoformat() if _state["updated_at"] else None
        ),
        "price_col": _state["price_col"],
    }


@app.post(
    "/sync_data",
    summary="接收企业系统推送的物料成本数据",
    tags=["数据同步"],
    dependencies=[Depends(_verify_token)],
)
async def sync_data(body: SyncRequest) -> dict:
    """
    接收 Java / 企业系统推送的批量数据并写入内存缓存与 PostgreSQL 表。

    - **mode=full**：全量替换内存缓存及 core_cost_records 表。
    - **mode=incremental**：追加后按（物料编码, 工厂, monitor_date）去重，保留最新记录。

    `records` 中的字段名可以是中文标准列名，
    也可以是 `processor.FIELD_MAP` 中定义的英文 / Java 字段名（如 `partId`、`validDate`）。
    """
    if not body.records:
        raise HTTPException(status_code=400, detail="records 不能为空")

    loop = asyncio.get_running_loop()

    # CPU-bound 数据处理放到线程池，避免阻塞异步事件循环
    df, price_col, err = await loop.run_in_executor(
        _executor,
        processor.process_records_from_json,
        body.records,
    )

    if err:
        raise HTTPException(status_code=422, detail=f"数据处理失败: {err}")

    if body.mode == "full" or _state["df"] is None:
        _state["df"] = df
    else:
        _state["df"] = (
            pd.concat([_state["df"], df], ignore_index=True)
            .drop_duplicates(subset=["物料编码", "工厂", "monitor_date"], keep="last")
            .reset_index(drop=True)
        )

    _state["price_col"] = price_col
    _state["updated_at"] = datetime.now()

    # 异步持久化（不阻塞响应返回）
    await loop.run_in_executor(_executor, _persist_cache, _state["df"], _state["price_col"])

    return {
        "status": "ok",
        "mode": body.mode,
        "received": len(body.records),
        "cached_total": int(len(_state["df"])),
        "price_col": price_col,
        "updated_at": _state["updated_at"].isoformat(),
    }


@app.post(
    "/detect_anomalies",
    summary="即时异常成本检测（不写入缓存）",
    tags=["分析"],
    dependencies=[Depends(_verify_token)],
)
async def detect_anomalies_endpoint(body: AnomalyRequest) -> dict:
    """
    接收数据并直接返回异常检测结果（JSON），为纯计算接口，**不修改内存缓存**。

    返回字段：
    - `total`: 检测的总记录数
    - `anomaly_count`: 异常记录数
    - `price_col`: 本次检测使用的价格列名
    - `results`: 每条记录的详细检测结果列表（含预测值、上下限、偏离比例、状态）
    """
    if not body.records:
        raise HTTPException(status_code=400, detail="records 不能为空")

    loop = asyncio.get_running_loop()

    df, price_col, err = await loop.run_in_executor(
        _executor,
        processor.process_records_from_json,
        body.records,
    )
    if err:
        raise HTTPException(status_code=422, detail=f"数据处理失败: {err}")

    # 允许调用方覆盖自动检测到的价格列
    if body.price_col:
        price_col = body.price_col

    try:
        result_df = await loop.run_in_executor(
            _executor,
            processor.detect_cost_anomalies,
            df,
            price_col,
        )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("异常成本检测失败")
        raise HTTPException(status_code=500, detail=f"检测失败: {exc}")

    # 序列化处理：把 numpy/pandas 类型转换为 Python 原生类型，确保 JSON 可序列化
    if "价格有效于" in result_df.columns:
        result_df["价格有效于"] = result_df["价格有效于"].astype(str)

    for col in ["实际成本", "预测值", "合理下限", "合理上限", "偏离数值"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(
                lambda x: round(float(x), 4) if pd.notna(x) else None
            )

    if "偏离比例" in result_df.columns:
        result_df["偏离比例"] = result_df["偏离比例"].apply(
            lambda x: round(float(x), 6) if pd.notna(x) else None
        )

    return {
        "total": len(result_df),
        "anomaly_count": int(
            result_df["status"].astype(str).str.contains("异常").sum()
        ),
        "price_col": price_col,
        "results": result_df.to_dict(orient="records"),
    }

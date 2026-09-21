"""FastAPI 应用：时间序列异常检测 + 趋势扫描 + 时滞关联接口。"""

from __future__ import annotations

import math
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import database
from .detection import analyze_series
from .lag_correlation import scan_lag_correlation
from .models import (
    DetectRequest,
    DetectResponse,
    DetectionListResponse,
    HealthResponse,
    LagCorrelationRequest,
    LagCorrelationResponse,
    TrendRequest,
    TrendResponse,
    parse_timestamp,
)
from .trends import scan_trends


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    database.init_db()
    yield


app = FastAPI(
    title="时间序列异常检测、趋势扫描与时滞关联 API",
    description=(
        "接收带时间戳的数值序列：异常检测基于稳健滑动窗口（中位数 + MAD）识别突变，"
        "返回合并后的异常区间、每段基线 / 分数 / 触发点解释，并将结果存入 SQLite；"
        "趋势扫描以滑窗 Mann-Kendall 检验 + Theil-Sen 斜率识别显著上升 / 下降趋势区间；"
        "时滞关联对双序列按候选时滞平移 driver 后做一对一最近邻匹配并计算 Pearson 相关。"
    ),
    version="1.2.0",
    lifespan=lifespan,
)


def _json_safe(value: Any) -> Any:
    """把非有限浮点（NaN/Infinity，可能来自请求 JSON）等不可序列化对象转为字符串。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: object, exc: RequestValidationError) -> JSONResponse:
    """422 处理器：清洗错误明细中的非有限浮点，避免回显 NaN/Infinity 时二次崩溃。"""
    return JSONResponse(
        status_code=422,
        content={"detail": [_json_safe(err) for err in exc.errors()]},
    )


def _prepare_series(points: list, label: str = "序列") -> tuple[list[str], list[float]]:
    """解析时间戳，按时间排序并检查重复。"""
    parsed = []
    for p in points:
        dt = parse_timestamp(p.timestamp)
        parsed.append((dt, p.value))
    parsed.sort(key=lambda item: item[0])

    timestamps: list[str] = []
    values: list[float] = []
    for dt, value in parsed:
        iso = dt.isoformat()
        if timestamps and iso == timestamps[-1]:
            raise HTTPException(status_code=422, detail=f"{label}存在重复时间戳: {iso}")
        timestamps.append(iso)
        values.append(value)
    return timestamps, values


def _strip_points(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "points"}


@app.post("/api/v1/detections", response_model=DetectResponse, status_code=201, tags=["检测"])
def create_detection(req: DetectRequest) -> dict:
    """提交序列进行异常检测，结果持久化后返回（含检测 ID）。"""
    timestamps, values = _prepare_series(req.points)

    result = analyze_series(
        timestamps=timestamps,
        values=values,
        window_size=req.window_size,
        min_window=req.min_window,
        threshold=req.threshold,
        merge_gap=req.merge_gap,
        baseline_policy=req.baseline_policy,
        rebase_after=req.rebase_after,
        gap_reset_factor=req.gap_reset_factor,
    )

    detection_id = uuid.uuid4().hex
    request_params = {
        "window_size": req.window_size,
        "min_window": req.min_window,
        "threshold": req.threshold,
        "merge_gap": req.merge_gap,
        "baseline_policy": req.baseline_policy,
        "rebase_after": req.rebase_after,
        "gap_reset_factor": req.gap_reset_factor,
        "point_count": len(req.points),
    }
    created_at = database.insert_detection(detection_id, req.series_id, request_params, result)

    body = {"id": detection_id, "series_id": req.series_id, "created_at": created_at, **result}
    if not req.include_points:
        body = {k: v for k, v in body.items() if k != "points"}
    return body


@app.get("/api/v1/detections/{detection_id}", response_model=DetectResponse, tags=["检测"])
def get_detection(
    detection_id: str,
    include_points: bool = Query(
        default=True, description="是否返回逐点注解（区间、基线、分数始终返回）"
    ),
) -> dict:
    """按 ID 查询历史检测结果。"""
    record = database.get_detection(detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"检测结果不存在: {detection_id}")
    result = record["result"]
    if not include_points:
        result = _strip_points(result)
    return {
        "id": record["id"],
        "series_id": record["series_id"],
        "created_at": record["created_at"],
        **result,
    }


@app.get("/api/v1/detections", response_model=DetectionListResponse, tags=["检测"])
def list_detections(
    series_id: str | None = Query(default=None, description="按业务序列 ID 过滤"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """列出历史检测（摘要信息），可按 series_id 过滤。"""
    items, total = database.list_detections(series_id=series_id, limit=limit, offset=offset)
    return {"items": items, "total": total}


@app.post("/api/v1/trends", response_model=TrendResponse, tags=["趋势"])
def create_trend_scan(req: TrendRequest) -> dict:
    """趋势扫描：滑窗 Mann-Kendall + Theil-Sen，合并显著趋势区间（无状态，不入库）。"""
    timestamps, values = _prepare_series(req.points)

    result = scan_trends(
        timestamps=timestamps,
        values=values,
        window_size=req.window_size,
        step_size=req.step_size,
        alpha=req.alpha,
        min_slope=req.min_slope,
    )

    body = {"series_id": req.series_id, **result}
    if not req.include_windows:
        body = {k: v for k, v in body.items() if k != "windows"}
    return body


@app.post(
    "/api/v1/lag-correlation",
    response_model=LagCorrelationResponse,
    tags=["时滞关联"],
)
def create_lag_correlation(req: LagCorrelationRequest) -> dict:
    """时滞关联扫描：对每个候选 lag 平移 driver 后与 response 做一对一最近邻匹配并计算 Pearson 相关（无状态）。"""
    driver_ts, driver_values = _prepare_series(req.driver_points, label="driver_points ")
    response_ts, response_values = _prepare_series(req.response_points, label="response_points ")

    return scan_lag_correlation(
        driver_timestamps=driver_ts,
        driver_values=driver_values,
        response_timestamps=response_ts,
        response_values=response_values,
        lag_seconds_list=req.lag_seconds,
        match_tolerance_seconds=req.match_tolerance_seconds,
        min_pairs=req.min_pairs,
    )


@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health() -> dict:
    return {"status": "ok", "database": database.DB_PATH}

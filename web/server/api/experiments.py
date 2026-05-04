# web/server/api/experiments.py
"""实验追踪 API — 列表、详情、对比、统计"""
from fastapi import APIRouter, Query, Body
from typing import Optional

from experiments.recorder import load_index, load_experiment

router = APIRouter()
BASE_DIR = "output"


@router.get("/")
def list_experiments(
    ranker: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    sort: str = Query("timestamp"),
    order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """实验列表，支持筛选、排序、分页。"""
    index = load_index(BASE_DIR)

    if ranker:
        index = [e for e in index if e.get("ranker") == ranker]
    if type:
        index = [e for e in index if e.get("type") == type]

    reverse = order == "desc"
    if sort.startswith("metrics."):
        metric_key = sort.split(".", 1)[1]
        index.sort(key=lambda e: e.get("metrics", {}).get(metric_key, 0) or 0, reverse=reverse)
    elif sort == "timestamp":
        index.sort(key=lambda e: e.get("timestamp", ""), reverse=reverse)
    elif sort == "factor_count":
        index.sort(key=lambda e: e.get("factor_count", 0), reverse=reverse)

    total = len(index)
    page = index[offset:offset + limit]

    return {"total": total, "items": page}


@router.get("/ranker/stats")
def ranker_stats(metric: str = Query("sharpe")):
    """按 ranker 分组统计。"""
    index = load_index(BASE_DIR)
    groups = {}
    for e in index:
        r = e.get("ranker", "Unknown")
        if r not in groups:
            groups[r] = []
        val = e.get("metrics", {}).get(metric)
        if val is not None:
            groups[r].append(val)

    return {
        "metric": metric,
        "groups": {r: {"count": len(vals), "mean": sum(vals)/len(vals) if vals else 0, "max": max(vals) if vals else 0}
                    for r, vals in groups.items()}
    }


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    """单个实验完整详情。"""
    exp = load_experiment(exp_id, BASE_DIR)
    if not exp:
        return {"error": "not found"}
    return exp


@router.post("/compare")
def compare_experiments(body: dict = Body(...)):
    """对比多个实验。返回指标对比 + 参数对比 + 差异标记。"""
    ids = body.get("ids", [])
    experiments = [load_experiment(eid, BASE_DIR) for eid in ids]
    experiments = [e for e in experiments if e is not None]
    if not experiments:
        return {"error": "no valid experiments"}

    all_metrics = set()
    for e in experiments:
        all_metrics.update(e.get("metrics", {}).keys())
    metrics_table = []
    for key in sorted(all_metrics):
        row = {"metric": key, "values": []}
        for e in experiments:
            row["values"].append(e.get("metrics", {}).get(key))
        row["has_diff"] = len(set(str(v) for v in row["values"] if v is not None)) > 1
        metrics_table.append(row)

    all_params = set()
    for e in experiments:
        all_params.update(e.get("config", {}).keys())
    params_table = []
    for key in sorted(all_params):
        row = {"param": key, "values": []}
        for e in experiments:
            row["values"].append(e.get("config", {}).get(key))
        row["has_diff"] = len(set(str(v) for v in row["values"] if v is not None)) > 1
        params_table.append(row)

    return {
        "experiments": [{k: e.get(k) for k in ["experiment_id", "timestamp", "type", "ranker", "date_range", "ranker_config"]} for e in experiments],
        "metrics_table": metrics_table,
        "params_table": params_table,
    }

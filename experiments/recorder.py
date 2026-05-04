# experiments/recorder.py
"""实验记录器 — 自动记录每次回测/优化的参数与指标"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = "experiments"
INDEX_FILE = "index.json"


def _exp_dir(base_dir: str) -> Path:
    p = Path(base_dir) / EXPERIMENTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


_id_counter = 0


def _make_id(ts: datetime) -> str:
    global _id_counter
    _id_counter += 1
    return f"exp_{ts.strftime('%Y%m%d_%H%M%S')}_{_id_counter:03d}"


def record_experiment(
    data: Dict[str, Any],
    base_dir: str = "output",
) -> str:
    """记录一次实验。自动生成 ID 和时间戳，写入 JSON 并更新索引。

    Args:
        data: {type, ranker, date_range, metrics, config, ...}
        base_dir: 实验存储根目录

    Returns:
        experiment_id
    """
    exp_id = _make_id(datetime.now())
    data["experiment_id"] = exp_id
    data["timestamp"] = datetime.now().isoformat()

    exp_dir = _exp_dir(base_dir)

    exp_path = exp_dir / f"{exp_id}.json"
    with open(exp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    index = load_index(base_dir)
    summary = {
        "experiment_id": exp_id,
        "timestamp": data["timestamp"],
        "type": data.get("type", ""),
        "ranker": data.get("ranker", ""),
        "date_range": data.get("date_range", {}),
        "metrics": data.get("metrics", {}),
        "factor_count": data.get("ranker_config", {}).get("factor_count", 0),
    }
    index.insert(0, summary)
    index = index[:200]
    with open(exp_dir / INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    logger.info(f"实验已记录: {exp_id} (ranker={data.get('ranker')}, sharpe={data.get('metrics', {}).get('sharpe', 'N/A')})")
    return exp_id


def load_index(base_dir: str = "output") -> List[Dict]:
    """加载实验索引列表（按时间倒序）。文件损坏或不存在时返回空列表。"""
    path = _exp_dir(base_dir) / INDEX_FILE
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning(f"索引文件损坏，将重建: {path}")
        return []


def load_experiment(exp_id: str, base_dir: str = "output") -> Optional[Dict]:
    """加载单个实验的完整记录。"""
    path = _exp_dir(base_dir) / f"{exp_id}.json"
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

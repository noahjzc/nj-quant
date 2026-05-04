# 实验追踪系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将实验追踪集成到 Web 看板，自动记录每次回测/优化的参数与指标，支持列表浏览、排序、多选对比（差异高亮）。

**Architecture:** `experiments/recorder.py` 提供 `record_experiment()` 函数写 JSON 文件 → FastAPI `/api/experiments` 路由读文件 → React 前端 Table + 对比面板展示。

**Tech Stack:** Python (FastAPI, json), TypeScript (React 18, Ant Design 5, Axios)

---

## 文件结构

```
新增:
  experiments/__init__.py
  experiments/recorder.py                        # 实验记录器
  web/server/api/experiments.py                  # 后端 API
  web/frontend/src/pages/Experiments/index.tsx   # 前端页面

修改:
  web/server/main.py                             # 注册路由 (1行)
  web/frontend/src/App.tsx                       # 导航 + 路由 (~15行)
  backtesting/run_daily_rotation.py              # 回测结束记录 (~5行)
  optimization/optuna/run_daily_rotation_optimization.py  # 优化结束记录 (~5行)
```

---

### Task 1: experiments/recorder.py — 实验记录核心

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/recorder.py`
- Test: `tests/experiments/test_recorder.py`

- [ ] **Step 1.1: 写测试**

```python
# tests/experiments/test_recorder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import tempfile
import os
from experiments.recorder import record_experiment, load_index, load_experiment


def test_record_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, 'experiments')
        record_experiment({
            "type": "backtest",
            "ranker": "TemporalMLRanker",
            "date_range": {"start": "2024-01-01", "end": "2024-06-30"},
            "metrics": {"sharpe": 1.52, "annual_return": 0.18, "max_drawdown": 0.12},
            "config": {"max_positions": 5, "max_total_pct": 0.8},
        }, base_dir=tmpdir)

        index = load_index(tmpdir)
        assert len(index) == 1
        assert index[0]["ranker"] == "TemporalMLRanker"
        assert index[0]["metrics"]["sharpe"] == 1.52

        exp = load_experiment(index[0]["experiment_id"], tmpdir)
        assert exp["config"]["max_positions"] == 5


def test_record_multiple():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            record_experiment({
                "type": "backtest",
                "ranker": "MLRanker",
                "date_range": {"start": "2024-01-01", "end": "2024-06-30"},
                "metrics": {"sharpe": 1.0 + i * 0.1},
                "config": {},
            }, base_dir=tmpdir)

        index = load_index(tmpdir)
        assert len(index) == 3
        # 默认按时间倒序
        assert index[0]["metrics"]["sharpe"] > index[-1]["metrics"]["sharpe"]


def test_load_index_corrupted():
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, 'experiments')
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, 'index.json'), 'w') as f:
            f.write('corrupted')
        index = load_index(tmpdir)
        assert index == []
```

- [ ] **Step 1.2: 运行测试确认失败**

```bash
cd D:/workspace/code/mine/quant/nj-quant
.venv/Scripts/python.exe -m pytest tests/experiments/test_recorder.py -v
# Expected: 3 FAIL
```

- [ ] **Step 1.3: 实现 recorder.py**

```python
# experiments/__init__.py
from experiments.recorder import record_experiment, load_index, load_experiment
```

```python
# experiments/recorder.py
"""实验记录器 — 自动记录每次回测/优化的参数与指标"""
import json
import logging
import os
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


def _make_id(ts: datetime) -> str:
    return f"exp_{ts.strftime('%Y%m%d_%H%M%S')}"


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

    # 写单个实验文件
    exp_path = exp_dir / f"{exp_id}.json"
    with open(exp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    # 更新索引
    index = load_index(base_dir)
    # 摘要：只保留列表页需要的字段
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
    # 保留最近 200 条
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
```

- [ ] **Step 1.4: 运行测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/experiments/test_recorder.py -v
# Expected: 3 PASS
```

---

### Task 2: web/server/api/experiments.py — 后端 API

**Files:**
- Create: `web/server/api/experiments.py`

- [ ] **Step 2.1: 实现 API**

```python
# web/server/api/experiments.py
"""实验追踪 API — 列表、详情、对比、统计"""
from fastapi import APIRouter, Query, Body
from typing import List, Optional
import json
from pathlib import Path

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

    # 筛选
    if ranker:
        index = [e for e in index if e.get("ranker") == ranker]
    if type:
        index = [e for e in index if e.get("type") == type]

    # 排序
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

    # 提取指标
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

    # 提取参数差异
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
```

- [ ] **Step 2.2: 验证**

```bash
cd D:/workspace/code/mine/quant/nj-quant
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
from web.server.api.experiments import router
print('OK: experiments API imports')
"
```

---

### Task 3: web/server/main.py — 注册路由

- [ ] **Step 3.1: 修改 main.py**

```python
# web/server/main.py — 在现有 import 后加一行:
from web.server.api import signals, positions, data_browser, cron_status, experiments

# 在现有 app.include_router 调用的末尾加:
app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
```

---

### Task 4: 前端 — Experiments 页面

**Files:**
- Create: `web/frontend/src/pages/Experiments/index.tsx`

- [ ] **Step 4.1: 实现前端页面**

```tsx
// web/frontend/src/pages/Experiments/index.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Table, Select, Button, Modal, Tag, Space, DatePicker, message } from 'antd';
import { BarChartOutlined } from '@ant-design/icons';
import api from '../../utils/api';

interface Experiment {
  experiment_id: string;
  timestamp: string;
  type: string;
  ranker: string;
  date_range: { start: string; end: string };
  metrics: Record<string, number>;
  factor_count: number;
}

interface CompareRow {
  metric?: string;
  param?: string;
  values: (number | null)[];
  has_diff: boolean;
}

const METRIC_COLUMNS = [
  { key: 'sharpe', label: 'Sharpe' },
  { key: 'annual_return', label: '年化收益' },
  { key: 'max_drawdown', label: '最大回撤' },
  { key: 'calmar', label: 'Calmar' },
  { key: 'win_rate', label: '胜率' },
  { key: 'ic', label: 'IC' },
  { key: 'total_return', label: '总收益' },
];

const TYPE_MAP: Record<string, string> = { backtest: '回测', optimization_single: '单期优化', optimization_wf: 'Walk-Forward', sensitivity: '敏感性分析' };
const RANKER_MAP: Record<string, string> = { SignalRanker: 'Signal', MLRanker: 'LightGBM', TemporalMLRanker: 'Temporal+LightGBM' };

const ExperimentsPage: React.FC = () => {
  const [data, setData] = useState<Experiment[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareData, setCompareData] = useState<{ experiments: any[]; metrics_table: CompareRow[]; params_table: CompareRow[] } | null>(null);
  const [filters, setFilters] = useState({ ranker: '', type: '', sort: 'timestamp', order: 'desc' });
  const [page, setPage] = useState(1);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const params: any = { limit: 20, offset: (page - 1) * 20, sort: filters.sort, order: filters.order };
      if (filters.ranker) params.ranker = filters.ranker;
      if (filters.type) params.type = filters.type;
      const res = await api.get('/experiments/', { params });
      setData(res.data.items);
      setTotal(res.data.total);
    } catch (e) {
      message.error('加载实验列表失败');
    }
    setLoading(false);
  }, [page, filters]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleCompare = async () => {
    if (selected.length < 2) return;
    const res = await api.post('/experiments/compare', { ids: selected });
    setCompareData(res.data);
    setCompareOpen(true);
  };

  const formatPct = (v: number | null | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '-';
  const formatNum = (v: number | null | undefined) => v != null ? v.toFixed(4) : '-';

  const columns: any[] = [
    { title: 'ID', dataIndex: 'experiment_id', width: 140 },
    { title: '时间', dataIndex: 'timestamp', width: 160, render: (v: string) => v?.replace('T', ' ').substring(0, 16) },
    { title: '类型', dataIndex: 'type', width: 90, render: (v: string) => TYPE_MAP[v] || v },
    {
      title: '模型', dataIndex: 'ranker', width: 150,
      render: (v: string) => <Tag color={v === 'TemporalMLRanker' ? 'blue' : v === 'MLRanker' ? 'green' : 'default'}>{RANKER_MAP[v] || v}</Tag>
    },
    { title: 'Sharpe', dataIndex: ['metrics', 'sharpe'], width: 80, sorter: true, render: formatNum },
    { title: '年化收益', dataIndex: ['metrics', 'annual_return'], width: 90, sorter: true, render: formatPct },
    { title: '最大回撤', dataIndex: ['metrics', 'max_drawdown'], width: 90, sorter: true, render: formatPct },
    { title: 'Calmar', dataIndex: ['metrics', 'calmar'], width: 80, sorter: true, render: formatNum },
    { title: '胜率', dataIndex: ['metrics', 'win_rate'], width: 70, sorter: true, render: formatPct },
    { title: 'IC', dataIndex: ['metrics', 'ic'], width: 70, sorter: true, render: formatNum },
    { title: '因字数', dataIndex: 'factor_count', width: 60, sorter: true },
  ];

  return (
    <div>
      <h2>实验追踪</h2>

      <Space style={{ marginBottom: 16 }}>
        <Select placeholder="模型类型" allowClear style={{ width: 160 }}
          value={filters.ranker || undefined}
          onChange={(v) => setFilters({ ...filters, ranker: v || '' })}>
          <Select.Option value="">全部</Select.Option>
          <Select.Option value="TemporalMLRanker">Temporal+LightGBM</Select.Option>
          <Select.Option value="MLRanker">LightGBM</Select.Option>
          <Select.Option value="SignalRanker">Signal</Select.Option>
        </Select>

        <Select placeholder="实验类型" allowClear style={{ width: 130 }}
          value={filters.type || undefined}
          onChange={(v) => setFilters({ ...filters, type: v || '' })}>
          <Select.Option value="">全部</Select.Option>
          <Select.Option value="backtest">回测</Select.Option>
          <Select.Option value="optimization_single">单期优化</Select.Option>
          <Select.Option value="optimization_wf">Walk-Forward</Select.Option>
          <Select.Option value="sensitivity">敏感性分析</Select.Option>
        </Select>

        <Button type="primary" disabled={selected.length < 2} onClick={handleCompare}>
          对比选中 ({selected.length})
        </Button>
      </Space>

      <Table
        rowKey="experiment_id"
        rowSelection={{ selectedRowKeys: selected, onChange: (keys) => setSelected(keys as string[]) }}
        columns={columns}
        dataSource={data}
        loading={loading}
        pagination={{ total, pageSize: 20, current: page, onChange: (p) => setPage(p), showTotal: (t) => `共 ${t} 条` }}
        onChange={(_p, _f, sorter: any) => {
          if (sorter.field) setFilters({ ...filters, sort: sorter.field === 'sharpe' ? 'metrics.sharpe' : sorter.field === 'annual_return' ? 'metrics.annual_return' : sorter.field === 'max_drawdown' ? 'metrics.max_drawdown' : sorter.field === 'calmar' ? 'metrics.calmar' : sorter.field === 'win_rate' ? 'metrics.win_rate' : sorter.field === 'ic' ? 'metrics.ic' : 'factor_count', order: sorter.order === 'ascend' ? 'asc' : 'desc' });
        }}
        size="middle"
      />

      <Modal
        title="实验对比"
        open={compareOpen}
        onCancel={() => setCompareOpen(false)}
        width={900}
        footer={null}
      >
        {compareData && (
          <>
            <div style={{ marginBottom: 16 }}>
              {compareData.experiments.map((e: any, i: number) => (
                <Tag key={i} color="blue" style={{ marginBottom: 4 }}>
                  {e.experiment_id} — {RANKER_MAP[e.ranker] || e.ranker}
                </Tag>
              ))}
            </div>

            <h4>指标对比</h4>
            <Table
              dataSource={compareData.metrics_table}
              columns={[
                { title: '指标', dataIndex: 'metric', width: 120,
                  render: (v: string) => METRIC_COLUMNS.find(m => m.key === v)?.label || v },
                ...compareData.experiments.map((e: any, i: number) => ({
                  title: e.experiment_id.substring(4, 16),
                  dataIndex: 'values',
                  width: 100,
                  render: (_: any, row: CompareRow) => {
                    const val = row.values[i];
                    const isBest = row.metric !== 'max_drawdown' && typeof val === 'number' && typeof row.values[0] === 'number' && val === Math.max(...row.values.filter((v: any): v is number => typeof v === 'number'));
                    const isBestMDD = row.metric === 'max_drawdown' && typeof val === 'number' && typeof row.values[0] === 'number' && val === Math.min(...row.values.filter((v: any): v is number => typeof v === 'number'));
                    const style: React.CSSProperties = {
                      color: (isBest || isBestMDD) ? '#389e0d' : undefined,
                      fontWeight: (isBest || isBestMDD) ? 'bold' : undefined,
                    };
                    return <span style={style}>{typeof val === 'number' ? (row.metric.includes('return') || row.metric.includes('drawdown') || row.metric.includes('win_rate') ? formatPct(val) : formatNum(val)) : '-'}</span>;
                  },
                  onCell: (_: any, row: CompareRow) => ({ style: { backgroundColor: row.has_diff ? '#fff7e6' : undefined } }),
                })),
              ]}
              rowKey="metric"
              size="small"
              pagination={false}
              style={{ marginBottom: 24 }}
            />

            <h4>参数对比</h4>
            <Table
              dataSource={compareData.params_table}
              columns={[
                { title: '参数', dataIndex: 'param', width: 180 },
                ...compareData.experiments.map((e: any, i: number) => ({
                  title: e.experiment_id.substring(4, 16),
                  dataIndex: 'values',
                  width: 120,
                  render: (_: any, row: CompareRow) => String(row.values[i] ?? '-'),
                  onCell: (_: any, row: CompareRow) => ({ style: { backgroundColor: row.has_diff ? '#fff7e6' : undefined } }),
                })),
              ]}
              rowKey="param"
              size="small"
              pagination={false}
            />
          </>
        )}
      </Modal>
    </div>
  );
};

export default ExperimentsPage;
```

- [ ] **Step 4.2: 验证前端编译**

```bash
cd D:/workspace/code/mine/quant/nj-quant/web/frontend
npm run build 2>&1 | tail -5
# Expected: 无错误（或仅 warnings）
```

---

### Task 5: 修改 App.tsx — 导航 + 路由

**File:** `web/frontend/src/App.tsx`

- [ ] **Step 5.1: 修改 App.tsx**

在 import 区域加：
```tsx
import { BarChartOutlined } from '@ant-design/icons';
import Experiments from './pages/Experiments';
```

在 Routes 内加：
```tsx
<Route path="/experiments" element={<Experiments />} />
```

在 items 数组加：
```tsx
{ key: '/experiments', icon: <BarChartOutlined />, label: '实验追踪' },
```

---

### Task 6: 自动记录 — 回测脚本

**File:** `backtesting/run_daily_rotation.py`

- [ ] **Step 6.1: 在回测结束时自动记录**

在 `main()` 函数中，回测结果生成后（metrics 已计算），加：

```python
from experiments.recorder import record_experiment

# Determine ranker type
ranker_type = "Unknown"
if hasattr(engine, 'ranker'):
    r = engine.ranker
    if hasattr(r, 'encoder'):
        ranker_type = "TemporalMLRanker"
    elif hasattr(r, 'model'):
        ranker_type = "MLRanker"
    else:
        ranker_type = "SignalRanker"

record_experiment({
    "type": "backtest",
    "ranker": ranker_type,
    "date_range": {"start": args.start, "end": args.end},
    "metrics": {
        "sharpe": metrics.get('sharpe_ratio', 0),
        "annual_return": metrics.get('annual_return', 0),
        "max_drawdown": metrics.get('max_drawdown', 0),
        "calmar": metrics.get('calmar_ratio', 0),
        "win_rate": metrics.get('win_rate', 0),
    },
    "config": {
        "max_positions": engine.config.max_positions,
        "max_total_pct": engine.config.max_total_pct,
        "max_position_pct": engine.config.max_position_pct,
        "stop_loss_mult": engine.config.stop_loss_mult,
        "take_profit_mult": engine.config.take_profit_mult,
        "trailing_pct": engine.config.trailing_pct,
        "atr_period": engine.config.atr_period,
    },
    "ranker_config": {
        "factor_count": len(engine.ranker.required_features) if hasattr(engine.ranker, 'required_features') else 0,
    },
})
```

---

### Task 7: 自动记录 — 优化脚本

**File:** `optimization/optuna/run_daily_rotation_optimization.py`

- [ ] **Step 7.1: 在优化结束时自动记录**

在 `run_single_optimization()` 返回前：

```python
from experiments.recorder import record_experiment

ranker_type = "Unknown"
if ranker is not None:
    if hasattr(ranker, 'encoder'):
        ranker_type = "TemporalMLRanker"
    elif hasattr(ranker, 'model'):
        ranker_type = "MLRanker"

record_experiment({
    "type": "optimization_single",
    "ranker": ranker_type,
    "date_range": {"start": start_date, "end": end_date},
    "metrics": {
        "sharpe": best_sharpe,
    },
    "config": _config_to_dict(best_config),
    "ranker_config": {
        "factor_count": len(best_config.rank_factor_weights) if hasattr(best_config, 'rank_factor_weights') else 0,
    },
})
```

---

### Task 8: 端到端验证

- [ ] **Step 8.1: 运行 recorder 测试**

```bash
.venv/Scripts/python.exe -m pytest tests/experiments/test_recorder.py -v
# Expected: 3 PASS
```

- [ ] **Step 8.2: 验证 API 可用**

```bash
cd D:/workspace/code/mine/quant/nj-quant
.venv/Scripts/python.exe -c "
from experiments.recorder import record_experiment, load_index
# 造一条假实验
record_experiment({
    'type': 'backtest', 'ranker': 'MLRanker',
    'date_range': {'start': '2024-01-01', 'end': '2024-06-30'},
    'metrics': {'sharpe': 1.5, 'annual_return': 0.2, 'max_drawdown': 0.1, 'calmar': 2.0, 'win_rate': 0.6, 'ic': 0.05},
    'config': {'max_positions': 5, 'max_total_pct': 0.8},
})
print('实验记录完成')
index = load_index()
print(f'索引共 {len(index)} 条')
"
```

- [ ] **Step 8.3: 验证后端启动**

```bash
.venv/Scripts/python.exe -c "
from web.server.main import app
print('FastAPI app 包含 /experiments 路由:',
      any(r.path.startswith('/experiments') for r in app.routes))
"
# Expected: True
```

- [ ] **Step 8.4: 验证前端编译**

```bash
cd web/frontend && npm run build 2>&1 | tail -5
# Expected: no errors
```

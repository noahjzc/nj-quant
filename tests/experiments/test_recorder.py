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
        # Most recent first
        assert index[0]["metrics"]["sharpe"] > index[-1]["metrics"]["sharpe"]


def test_load_index_corrupted():
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, 'experiments')
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, 'index.json'), 'w') as f:
            f.write('corrupted')
        index = load_index(tmpdir)
        assert index == []

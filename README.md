# nj-quant

A-share quantitative trading system — daily rotation backtesting, multi-factor ranking, ML-based stock selection, Optuna hyperparameter optimization, and temporal feature learning.

## Features

- **Daily Rotation Engine** — Vectorized signal generation, two-phase trade execution, market regime detection
- **Alpha158 Factors** — 156 financial factors (KBar, Price ratios, Rolling operators across 5 time windows)
- **Factor Screening & Orthogonalization** — Rank IC analysis, Gram-Schmidt decorrelation, dual output (raw + orthogonal factor sets)
- **ML Ranking** — LightGBM model training and inference, Optuna hyperparameter optimization
- **Temporal Feature Layer** — Transformer Encoder pretrained via self-supervised masked prediction, combined with LightGBM for cross-sectional ranking
- **Optuna Optimization** — TPE-based Bayesian optimization, single-period and walk-forward modes, robustness-based selection
- **Robustness Testing** — Monte Carlo simulation, CSCV, deflated Sharpe, parameter sensitivity analysis
- **Web Dashboard** — FastAPI backend + Vite/React frontend for signal monitoring and data browsing

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd nj-quant
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
pip install -e ".[dev]"  # for testing

# Configure database
cp config/database.example.ini config/database.ini
# Edit database.ini with your PostgreSQL credentials

# Build data cache
python -c "
from data.cache.daily_data_cache import DailyDataCache
DailyDataCache.build('2023-01-01', '2024-12-31', 'cache/daily_rotation')
"

# Run a backtest
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31
```

## Optimization Pipeline

```bash
# Stage 0: Factor screening
python -m strategy.factors.factor_screening \
    --start 2020-01-01 --end 2022-12-31 --output output/

# Stage 1: Train ML ranker (with optional temporal encoder)
python -m strategy.ml.temporal.pretrain \
    --start 2020-01-01 --end 2022-12-31 \
    --factors output/selected_factors.json --epochs 50 --output output/

python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2022-12-31 \
    --factors output/selected_factors.json \
    --encoder output/temporal_encoder.pt --trials 50 --output output/

# Stage 2: Framework optimization (auto-loads best model)
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 \
    --ml-model auto --trials 100 --output output/
```

## Key CLI Commands

| Command | Description |
|---------|-------------|
| `nj-quant-backtest` | Run a single daily rotation backtest |
| `nj-quant-optimize` | Run framework parameter optimization (Optuna) |
| `nj-quant-ml-optimize` | Run ML hyperparameter optimization |
| `nj-quant-factor-screen` | Run factor screening and orthogonalization |
| `nj-quant-pretrain` | Pretrain TemporalEncoder (self-supervised) |

Or run directly:

| Script | Purpose |
|--------|---------|
| `python backtesting/run_daily_rotation.py` | Single backtest with analysis |
| `python optimization/optuna/run_daily_rotation_optimization.py` | Optuna framework optimization |
| `python optimization/optuna/run_ml_optimization.py` | ML model training + hyperparameter optimization |
| `python -m strategy.factors.factor_screening` | Factor IC analysis and orthogonalization |
| `python -m strategy.ml.temporal.pretrain` | Temporal Encoder self-supervised pretraining |

## Architecture

```
nj-quant/
├── backtesting/                  # Backtesting framework
│   ├── run_daily_rotation.py    # CLI entry + result export
│   ├── analysis/                # Performance analysis & visualization
│   ├── risk/                    # Risk management (ATR stops, position sizing)
│   └── costs/                   # Transaction cost models
├── strategy/                     # Strategy definitions
│   ├── rotation/                # Daily rotation engine (core)
│   │   ├── daily_rotation_engine.py
│   │   ├── config.py            # RotationConfig dataclass
│   │   └── signal_engine/       # Signal filter + ranker
│   ├── factors/                 # Factor computation
│   │   ├── alpha158.py          # 156 Alpha158 factors
│   │   └── factor_screening.py  # IC analysis + orthogonalization
│   └── ml/                      # ML ranking
│       ├── ml_ranker.py         # LightGBM inference
│       ├── trainer.py           # MLRankerTrainer
│       ├── ml_optuna.py         # Optuna-based ML hyperparameter optimization
│       └── temporal/            # Temporal feature layer
│           ├── encoder.py       # TemporalEncoder (Transformer)
│           ├── pretrain.py      # Self-supervised pretraining
│           ├── temporal_ranker.py    # TemporalMLRanker
│           └── temporal_trainer.py   # Phase 2 joint training
├── optimization/                 # Parameter optimization
│   └── optuna/                  # Optuna TPE optimization
├── data/                        # Data layer (PostgreSQL, Parquet cache)
├── robustness/                   # Robustness analysis
├── web/                         # FastAPI backend + React frontend
├── signal_pipeline/             # Live trading signals
├── docs/superpowers/            # Design specs + implementation plans
└── tests/                       # Test suite
```

## Testing

```bash
pytest tests/ -v                        # Run all tests
pytest tests/strategy/ml/temporal/ -v   # Temporal feature layer tests
pytest tests/robustness/ -v             # Robustness tests
```

## Tech Stack

| Category | Packages |
|----------|----------|
| ML | lightgbm, scikit-learn, joblib |
| Deep Learning | torch (TemporalEncoder) |
| Optimization | optuna, scipy |
| Data | pandas, numpy, sqlalchemy, psycopg2 |
| Data Sources | akshare, tushare |
| Web | fastapi, uvicorn, pydantic |
| Viz | matplotlib, seaborn |

Python ≥ 3.10. Full dependency list in `pyproject.toml` or `requirements.txt`.

## Design Docs

Key specifications in `docs/superpowers/specs/`:

- `2026-04-25-daily-rotation-design.md` — Daily rotation engine
- `2026-04-26-daily-rotation-optuna-design.md` — Optuna optimization
- `2026-04-30-robustness-and-cost-design.md` — Robustness analysis
- `2026-05-02-ml-optuna-integration-design.md` — ML + Optuna three-stage integration
- `2026-05-02-qlib-llm-research-and-roadmap.md` — Qlib/RD-Agent research
- `2026-05-03-temporal-feature-layer-design.md` — Temporal feature layer

## Configuration

Database connection in `config/database.ini`:

```ini
[postgresql]
host = localhost
port = 5432
database = your_db
user = your_user
password = your_pass
```

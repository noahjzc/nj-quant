# Trial Early Termination via Optuna Pruning

**Date:** 2026-04-29
**Status:** Approved

## Problem

回测中资产跌到初始资本的 50% 以下时，trial 仍跑完全部 1455 天，浪费计算资源。当前 `MAX_DRAWDOWN_LIMIT` 检查在 `engine.run()` 全部结束后才执行。

## Design

### 1. Config: `min_asset_ratio` (RotationConfig)

```python
min_asset_ratio: float = 0.5  # 资产跌到初始资本的该比例以下时触发淘汰
```

### 2. Engine: `run(trial=None)` 每日检查

`run()` 新增可选参数 `trial: optuna.Trial = None`。每日循环末尾：

```python
if result.total_asset < self.config.initial_capital * self.config.min_asset_ratio:
    if trial is not None:
        trial.report(result.total_asset / self.config.initial_capital, i)
        raise optuna.TrialPruned(
            f"资产 {result.total_asset:,.0f} < 初始 {self.config.initial_capital * self.config.min_asset_ratio:,.0f}"
        )
    else:
        return self.daily_results  # 非 Optuna 场景，提前返回
```

### 3. objective(): 传 trial 给 engine

```python
results = engine.run(trial=trial)
```

`TrialPruned` 异常由 Optuna 自动捕获，无需修改 objective 的异常处理。

## Files Changed

| File | Change |
|------|--------|
| `back_testing/rotation/config.py` | +1 field `min_asset_ratio` |
| `back_testing/rotation/daily_rotation_engine.py` | `run()` 新增 `trial` 参数 + 每日检查 |
| `back_testing/optimization/run_daily_rotation_optimization.py` | `engine.run(trial=trial)` |

## Non-Optuna Behavior

不传 `trial` 时（单次回测 `run_daily_rotation.py`），触发阈值后直接提前返回，行为安全。

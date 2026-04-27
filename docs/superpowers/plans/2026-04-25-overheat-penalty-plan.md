# OVERHEAT 过热度惩罚因子 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在多因子排序中新增 OVERHEAT 惩罚因子，对 RSI 超买 + 短期涨幅双高的股票降权，减少追高买入。

**Architecture:** 在 `RotationConfig` 新增阈值和因子配置，在 `_execute_buy` 因子构建循环中计算 RET_5 和 OVERHEAT，复用现有 `SignalRanker` 的 z-score + 方向反转机制。新增 `compute_overheat()` 函数便于测试。

**Tech Stack:** Python 3.12, pandas, dataclasses

---

## 文件结构

| 文件 | 改动类型 | 职责 |
|------|---------|------|
| `back_testing/rotation/daily_rotation_engine.py` | 修改 | 新增 `compute_overheat()` 函数；`_execute_buy` 中计算 RET_5 和 OVERHEAT |
| `back_testing/rotation/config.py` | 修改 | 新增阈值字段；因子权重/方向追加 OVERHEAT |
| `tests/back_testing/rotation/test_overheat.py` | 创建 | 测试 `compute_overheat()` 各种场景 |

---

### Task 1: 新增 `compute_overheat()` 函数并编写测试

**Files:**
- Create: `tests/back_testing/rotation/test_overheat.py`
- Modify: `back_testing/rotation/daily_rotation_engine.py`

- [ ] **Step 1: 编写测试文件**

```python
"""测试 OVERHEAT 过热度计算"""
import pytest
from back_testing.rotation.daily_rotation_engine import compute_overheat


class TestComputeOverheat:
    def test_no_overheat_when_below_thresholds(self):
        """RSI 和 RET_5 都在阈值以下，返回 0"""
        assert compute_overheat(rsi=70, ret5=0.10) == 0.0
        assert compute_overheat(rsi=80, ret5=0.10) == 0.0  # RSI 超标但涨幅不够
        assert compute_overheat(rsi=70, ret5=0.20) == 0.0  # 涨幅超标但 RSI 不够

    def test_overheat_when_both_above_thresholds(self):
        """双高时返回正数过热度"""
        result = compute_overheat(rsi=80, ret5=0.20)
        assert result > 0.0

    def test_overheat_max_value(self):
        """极端过热接近 1.0"""
        result = compute_overheat(rsi=100, ret5=0.50)
        assert 0.9 <= result <= 1.0

    def test_overheat_zero_at_exact_threshold(self):
        """恰好在阈值时过热度为 0（不触发）"""
        assert compute_overheat(rsi=75, ret5=0.20) == 0.0  # RSI 不满足 >
        assert compute_overheat(rsi=80, ret5=0.15) == 0.0  # RET5 不满足 >

    def test_custom_thresholds(self):
        """自定义阈值生效"""
        result = compute_overheat(rsi=80, ret5=0.10, rsi_threshold=80, ret5_threshold=0.05)
        assert result == 0.0  # RSI 不满足 >

    def test_negative_ret5(self):
        """负涨幅不触发过热"""
        assert compute_overheat(rsi=85, ret5=-0.10) == 0.0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/back_testing/rotation/test_overheat.py -v`
Expected: FAIL — `compute_overheat` not defined

- [ ] **Step 3: 在 `daily_rotation_engine.py` 中实现函数**

在文件顶部（import 之后、dataclass 之前）添加：

```python
def compute_overheat(
    rsi: float,
    ret5: float,
    rsi_threshold: float = 75.0,
    ret5_threshold: float = 0.15
) -> float:
    """计算过热度（0~1）。仅当 RSI 和短期涨幅双高时返回正值。"""
    if not (rsi > rsi_threshold and ret5 > ret5_threshold):
        return 0.0

    rsi_component = max(0.0, (rsi - rsi_threshold) / (100 - rsi_threshold))
    ret_component = min(1.0, max(0.0, (ret5 - ret5_threshold) / 0.35))
    return (rsi_component + ret_component) / 2.0
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/back_testing/rotation/test_overheat.py -v`
Expected: 6 PASS

- [ ] **Step 5: 提交**

```bash
git add tests/back_testing/rotation/test_overheat.py back_testing/rotation/daily_rotation_engine.py
git commit -m "feat: add compute_overheat() for overheat penalty factor"
```

---

### Task 2: 更新 RotationConfig 配置

**Files:**
- Modify: `back_testing/rotation/config.py`

- [ ] **Step 1: 添加过热度阈值字段**

在 `RotationConfig` 的 ATR 参数区（line 72 附近）之前插入：

```python
    # 过热度惩罚
    overheat_rsi_threshold: float = 75.0     # RSI 超买阈值
    overheat_ret5_threshold: float = 0.15    # 5日涨幅阈值
```

- [ ] **Step 2: 更新 rank_factor_weights**

在 `rank_factor_weights` 字典末尾追加：

```python
        'OVERHEAT': 0.20,
```

完整结果：
```python
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'RSI_1': 0.20,
        'RET_20': 0.15,
        'VOLUME_RATIO': 0.15,
        'PB': 0.25,
        'PE_TTM': 0.25,
        'OVERHEAT': 0.20,
    })
```

- [ ] **Step 3: 更新 rank_factor_directions**

追加 `'OVERHEAT': -1`：
```python
    rank_factor_directions: Dict[str, int] = field(default_factory=lambda: {
        'RSI_1': 1,
        'RET_20': 1,
        'VOLUME_RATIO': 1,
        'PB': -1,
        'PE_TTM': -1,
        'OVERHEAT': -1,
    })
```

- [ ] **Step 4: 验证导入**

Run: `python -c "from back_testing.rotation.config import RotationConfig; c = RotationConfig(); print(c.overheat_rsi_threshold, c.overheat_ret5_threshold, 'OVERHEAT' in c.rank_factor_weights)"`
Expected: `75.0 0.15 True`

- [ ] **Step 5: 提交**

```bash
git add back_testing/rotation/config.py
git commit -m "feat: add OVERHEAT factor to RotationConfig"
```

---

### Task 3: 在 _execute_buy 中集成 RET_5 和 OVERHEAT 计算

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:362-374`

- [ ] **Step 1: 在因子构建循环中新增 RET_5 和 OVERHEAT 处理**

将 `_execute_buy` 中的因子构建循环（line 362-373）替换为：

```python
            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    if len(df) >= 20 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'RET_5':
                    # 5日收益率（用于 OVERHEAT 计算）
                    if len(df) >= 5 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-5] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'OVERHEAT':
                    rsi_val = row.get('rsi_1', np.nan)
                    ret5_val = factor_row.get('RET_5', 0.0)
                    if pd.notna(rsi_val) and pd.notna(ret5_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), float(ret5_val),
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor in row.index:
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan
                else:
                    factor_row[factor] = np.nan
```

注意：`RET_5` 必须在循环中排在 `OVERHEAT` 之前（因为 `factor_row.get('RET_5', ...)` 依赖它）。由于 `factor_weights` 是字典（Python 3.7+ 保持插入顺序），需确保 `rank_factor_weights` 中 `'RET_5'` 在 `'OVERHEAT'` 之前。

**但**当前 `RET_5` 不在 `rank_factor_weights` 中——它只是 OVERHEAT 的中间计算量，不需要参与排序。所以用局部变量存 RET_5 即可：

修正方案：在进入因子循环前计算 RET_5，循环中只处理 OVERHEAT。

```python
            # 计算 RET_5（OVERHEAT 计算需要）
            if len(df) >= 5 and 'close' in df.columns:
                ret5 = row['close'] / df['close'].iloc[-5] - 1
            else:
                ret5 = 0.0

            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    if len(df) >= 20 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'OVERHEAT':
                    rsi_val = row.get('rsi_1', np.nan)
                    if pd.notna(rsi_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), ret5,
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor in row.index:
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan
                else:
                    factor_row[factor] = np.nan
```

- [ ] **Step 2: 语法验证**

Run: `python -c "from back_testing.rotation.daily_rotation_engine import DailyRotationEngine, compute_overheat; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 运行回测验证**

Run: `PYTHONPATH=. python back_testing/backtest/run_daily_rotation.py --start 2024-01-01 --end 2024-03-31`
Expected: 正常运行无报错，[TOP] 日志中过热股排名靠后

- [ ] **Step 4: 运行全部测试**

Run: `pytest tests/back_testing/rotation/test_overheat.py tests/back_testing/ -v`
Expected: 新增 6 个测试 PASS，已有测试无新增失败

- [ ] **Step 5: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat: integrate OVERHEAT penalty into stock ranking"
```

---

## 验证清单

- [x] `compute_overheat(rsi=70, ret5=0.10)` → 0.0（均不触发）
- [x] `compute_overheat(rsi=80, ret5=0.10)` → 0.0（涨幅不足）
- [x] `compute_overheat(rsi=70, ret5=0.20)` → 0.0（RSI 不足）
- [x] `compute_overheat(rsi=80, ret5=0.20)` → > 0（双高触发）
- [x] `compute_overheat(rsi=100, ret5=0.50)` → ≈ 1.0（极端过热）
- [x] `RotationConfig()` 创建时包含 OVERHEAT 因子，权重 0.20，方向 -1
- [x] 回测运行无报错
- [x] 已有测试无回归

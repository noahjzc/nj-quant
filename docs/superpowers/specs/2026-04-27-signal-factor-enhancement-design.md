# 信号与因子增强设计

Date: 2026-04-27

## 目标

利用已有数据库中未使用的列（circulating_mv、psy/psyma、kdj_j 等），从 OHLCV 推导新因子（WR），提升每日轮动策略的信号质量和排名效果。

## 范围

- 新增 3 个信号类型：KDJ_GOLD_LOW、PSY_BUY、PSY_SELL
- 新增 3 个排名因子：circulating_mv（对数化）、WR_10、WR_14
- 废弃信号清理：DMI_GOLD/DMI_DEATH 从默认配置移除（数据库缺少 dmi_plus_di/dmi_minus_di 列，永不触发）
- Optuna 优化参数扩展：新增因子权重 + KDJ 低位阈值
- 引擎层向量化扩展：PSY 和 KDJ_GOLD_LOW 纳入批量信号检测

## 不变

- DMI 枚举和检测器代码保留不动（避免破坏性 API 变更）
- 现有 6 因子和 7 买入信号的逻辑不修改
- Ranker 归一化逻辑不变（除以 total_weight）

---

## 一、数据结构变更

### 1.1 SignalType 新增枚举 (`base_signal.py`)

```python
# 买入信号
KDJ_GOLD_LOW = 'KDJ_GOLD_LOW'    # KDJ 金叉 + K < kdj_low_threshold
PSY_BUY = 'PSY_BUY'               # PSY < 25 且 PSY > PSYMA（超卖 + 趋势确认）

# 卖出信号
PSY_SELL = 'PSY_SELL'             # PSY > 75 且 PSY < PSYMA（超买 + 趋势确认）
```

`is_buy` property 需更新：KDJ_GOLD_LOW 和 PSY_BUY 归为买入。

### 1.2 RotationConfig 默认值变更 (`config.py`)

```python
# buy_signal_types 新增
"KDJ_GOLD_LOW"

# sell_signal_types 移除 DMI_DEATH，新增 PSY_SELL
# 新默认：['KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
#           'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN', 'PSY_SELL']

# 新增 KDJ 低位阈值
kdj_low_threshold: float = 30.0

# rank_factor_weights 新增（初始权重，优化后会调整）
"circulating_mv": 0.15
"WR_10": 0.10
"WR_14": 0.10

# rank_factor_directions 新增
'circulating_mv': -1   # 小市值好
'WR_10': -1            # WR 越小（越超卖）越好
'WR_14': -1
```

---

## 二、信号检测器变更 (`signal_filter.py`)

### 2.1 KDJGoldLowSignal

```python
class KDJGoldLowSignal(BaseSignal):
    def __init__(self, k_threshold: float = 30.0):
        super().__init__(SignalType.KDJ_GOLD_LOW)
        self.k_threshold = k_threshold

    def detect(self, df, stock_code):
        k = df['kdj_k']; d = df['kdj_d']
        triggered = self._cross_up(k, d) and k.iloc[-1] < self.k_threshold
        return SignalResult(...)
```

### 2.2 PSYBuySignal

```python
class PSYBuySignal(BaseSignal):
    def detect(self, df, stock_code):
        psy_now = df['psy'].iloc[-1]; psyma_now = df['psyma'].iloc[-1]
        triggered = psy_now < 25 and psy_now > psyma_now
        return SignalResult(...)
```

### 2.3 PSYSellSignal

```python
class PSYSellSignal(BaseSignal):
    def detect(self, df, stock_code):
        psy_now = df['psy'].iloc[-1]; psyma_now = df['psyma'].iloc[-1]
        triggered = psy_now > 75 and psy_now < psyma_now
        return SignalResult(...)
```

### 2.4 SignalFilter 更新

- `_SIGNAL_MAP` 注册三个新映射
- `__init__` 接受 `kdj_low_threshold` 参数，传入 `KDJGoldLowSignal`

---

## 三、新因子计算 (`factor_utils.py`)

### 3.1 WR 威廉指标

```python
@staticmethod
def williams_r(df: pd.DataFrame, period: int) -> float:
    high_n = df['high'].tail(period).max()
    low_n = df['low'].tail(period).min()
    close = df['close'].iloc[-1]
    if high_n == low_n:
        return -50.0
    return (high_n - close) / (high_n - low_n) * -100
```

### 3.2 circulating_mv

直接从 DB 列读取，取 `log(circulating_mv)` 做对数变换，在 `_execute_buy` 因子提取阶段完成。

---

## 四、引擎层变更 (`daily_rotation_engine.py`)

### 4.1 SignalFilter 构造

```python
self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode,
                                kdj_low_threshold=config.kdj_low_threshold)
```

### 4.2 `_build_signal_features` 扩展

特征矩阵新增 `psy`、`psyma` 列（latest 和 prev 各一份）。`kdj_k` 已存在于现有特征矩阵中，无需额外添加。

### 4.3 `_scan_buy_candidates` 向量化扩展

新增两个向量化 mask：

```python
if 'KDJ_GOLD_LOW' in active_signals:
    k_thresh = self.config.kdj_low_threshold
    masks['KDJ_GOLD_LOW'] = (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p']) & (f['kdj_k'] < k_thresh)

if 'PSY_BUY' in active_signals:
    masks['PSY_BUY'] = (f['psy'] < 25) & (f['psy'] > f['psyma'])
```

### 4.4 `_execute_buy` 因子提取扩展

```python
if factor == 'circulating_mv':
    val = row.get('circulating_mv', np.nan)
    factor_row[factor] = np.log(val) if val > 0 else np.nan
elif factor in ('WR_10', 'WR_14'):
    period = 10 if factor == 'WR_10' else 14
    factor_row[factor] = FactorProcessor.williams_r(df, period)
```

---

## 五、Optuna 优化参数 (`run_daily_rotation_optimization.py`)

### 5.1 `sample_config()` 新增采样

| 参数 | 范围 | 类型 |
|------|------|------|
| `kdj_low_threshold` | 20.0 ~ 40.0 | suggest_float |
| `rank_factor_weights['circulating_mv']` | 0.05 ~ 0.30 | suggest_float |
| `rank_factor_weights['WR_10']` | 0.0 ~ 0.20 | suggest_float |
| `rank_factor_weights['WR_14']` | 0.0 ~ 0.20 | suggest_float |

### 5.2 权重归一化

独立采样每个因子权重，`SignalRanker._calculate_scores` 中已有 `除以 total_weight` 归一化，不需要额外改动。

---

## 六、测试

需要新增/更新的测试：

1. `test_signal_filter.py` — KDJ_GOLD_LOW 阈值行为、PSY_BUY/SELL 阈值 + 趋势确认
2. `test_factor_utils.py` — williams_r 计算正确性（含边界：high==low）
3. `test_multi_factor_selector.py` — 新因子权重参与排名

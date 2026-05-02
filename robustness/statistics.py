import numpy as np


def deflated_sharpe_ratio(daily_returns: np.ndarray, rf_annual: float = 0.025,
                          periods_per_year: int = 252, n_trials: int = 100) -> float:
    """Deflated Sharpe Ratio (Harvey & Liu 2015)
    考虑多重测试惩罚后的 Sharpe 显著性。
    DSR > 2: 大致对应 p < 0.05
    DSR > 3: 大致对应 p < 0.001
    """
    n = len(daily_returns)
    if n < 2:
        return 0.0

    rf_daily = rf_annual / periods_per_year
    excess = daily_returns - rf_daily
    std_excess = np.std(excess, ddof=1)
    if std_excess < 1e-10:
        return 0.0
    observed_sharpe = np.mean(excess) / std_excess * np.sqrt(periods_per_year)

    if n_trials <= 1:
        return float(observed_sharpe)

    # E[max Sharpe] under null hypothesis of no predictability
    e_max = np.sqrt(2 * np.log(n_trials))
    var_max = 1.0 / n

    if var_max < 1e-10:
        return 0.0

    dsr = (observed_sharpe - e_max) / np.sqrt(var_max)
    return float(dsr)


def probability_of_backtest_overfit(is_sharpes: np.ndarray, oos_sharpes: np.ndarray) -> float:
    """PBO: IS最佳解在OOS排名后50%的概率 (Bailey et al. 2014)
    计算 IS 高排名但 OOS 低排名的组合比例。
    """
    n = len(is_sharpes)
    if n == 0 or len(oos_sharpes) == 0:
        return 0.0

    # For each trial, count how many IS are better and how many OOS are worse
    below_median = 0
    median_rank = n / 2.0

    for i in range(n):
        is_rank = np.sum(is_sharpes > is_sharpes[i])
        oos_rank = np.sum(oos_sharpes > oos_sharpes[i])
        # Flag cases where IS ranks high but OOS ranks low
        if is_rank < median_rank and oos_rank >= median_rank:
            below_median += 1

    return below_median / n

"""
Walk-Forward: 滚动窗口分析 - 防止过拟合的核心方法

Walk-Forward Analysis (WFA)是量化交易中防止过拟合的关键技术:

问题:
- 用全量历史数据优化出的参数，在历史回测中表现很好
- 但实盘可能表现很差(过拟合)

原因:
- 参数在历史数据上"记忆"了行情特征
- 这些特征未来可能不再重复

解决方案 - Walk-Forward:
1. 用历史一段数据(训练期)优化参数
2. 用紧跟的下一段数据(验证期)测试
3. 用更后面的数据(测试期)最终评估
4. 滚动向前，重复上述过程

窗口结构:
| 阶段   | 时间跨度 | 用途                    |
|--------|----------|------------------------|
| 训练期  | 3年     | GA搜索最优因子权重         |
| 验证期  | 1年     | 早停/选择最优个体          |
| 测试期  | 1年     | 最终绩效评估(样本外)        |

滚动步进: 每3个月向前滚动一次，重新优化

示例:
第1轮: 2019-01~2022-01(训练) → 2022-01~2023-01(验证) → 2023-01~2024-01(测试)
第2轮: 2019-04~2022-04(训练) → 2022-04~2023-04(验证) → 2023-04~2024-04(测试)
...
"""
import pandas as pd
from typing import List, Dict, Tuple


class WalkForwardAnalyzer:
    """
    Walk-Forward窗口分析器

    负责:
    1. 生成滚动窗口
    2. 聚合多窗口的最优权重
    """

    def __init__(self,
                 train_window_years: int = 3,
                 val_window_years: int = 1,
                 test_window_years: int = 1,
                 step_months: int = 3):
        """
        初始化Walk-Forward分析器

        Args:
            train_window_years: 训练窗口长度(年)
                               足够长的训练期才能让GA找到好解
            val_window_years: 验证窗口长度(年)
                             用于早停判断和选择最优个体
            test_window_years: 测试窗口长度(年)
                              最终绩效评估，越长越可靠
            step_months: 滚动步进(月)
                        越小窗口越多，但每窗口数据越少
        """
        self.train_window_years = train_window_years
        self.val_window_years = val_window_years
        self.test_window_years = test_window_years
        self.step_months = step_months

    def get_windows(self, start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> List[Dict]:
        """
        生成Walk-Forward窗口列表

        从start_date开始，每3个月滚动一次，
        生成多个(train, val, test)三期窗口。

        每个窗口结构:
        {
            'train': (train_start, train_end),
            'val': (val_start, val_end),
            'test': (test_start, test_end)
        }

        Args:
            start_date: 数据开始日期
            end_date: 数据结束日期

        Returns:
            窗口列表
        """
        windows = []
        current = pd.Timestamp(start_date)

        # 转换为月数
        train_months = self.train_window_years * 12
        val_months = self.val_window_years * 12
        test_months = self.test_window_years * 12

        while True:
            # 计算各期结束日期
            train_end = current + pd.DateOffset(months=train_months)
            val_end = train_end + pd.DateOffset(months=val_months)
            test_end = val_end + pd.DateOffset(months=test_months)

            # 如果测试期超出数据范围，停止
            if test_end > end_date:
                break

            # 添加窗口
            windows.append({
                'train': (current, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end)
            })

            # 向前滚动step_months
            current = current + pd.DateOffset(months=self.step_months)

        return windows

    def aggregate_weights(self, weights_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合多窗口的最优权重

        策略: 简单平均
        - 对每个因子，取所有窗口最优权重的均值
        - 然后归一化使总和=1.0

        为什么用平均:
        - 不同窗口可能学到不同的市场特征
        - 平均可以平滑掉极端权重
        - 提高权重稳定性，减少过拟合

        Args:
            weights_list: 各窗口最优权重列表

        Returns:
            聚合后的权重字典
        """
        if not weights_list:
            return {}

        n = len(weights_list)

        # 收集所有因子
        aggregated = {}
        all_keys = set()
        for w in weights_list:
            all_keys.update(w.keys())

        # 对每个因子计算平均值
        for key in all_keys:
            values = [w.get(key, 0) for w in weights_list]
            aggregated[key] = sum(values) / n

        # 归一化
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return aggregated

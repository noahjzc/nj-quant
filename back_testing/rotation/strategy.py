"""AbstractRotationStrategy — 策略抽象接口"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional
from back_testing.rotation.config import RotationConfig


class AbstractRotationStrategy(ABC):
    """
    每日轮动策略抽象基类

    定义策略的核心接口，支持：
    - 独立运行（run 方法）
    - GA 适应度评估（fitness 方法）
    """

    @abstractmethod
    def run(self, start_date: str, end_date: str, config: Optional[RotationConfig] = None) -> pd.DataFrame:
        """
        运行策略回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            config: 策略配置

        Returns:
            每日净值 DataFrame，index=date, columns=['total_asset', 'cash', 'position_value']
        """
        pass

    @abstractmethod
    def fitness(self, genome: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """
        GA 适应度函数

        Args:
            genome: GA 参数字典
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            适应度分数（Sharpe Ratio）
        """
        pass

    @abstractmethod
    def get_performance(self, results_df: pd.DataFrame) -> Dict:
        """
        计算绩效指标

        Returns:
            绩效字典 {sharpe, annual_return, max_drawdown, ...}
        """
        pass

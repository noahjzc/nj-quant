"""
Sensitivity: 敏感性分析 - 评估因子重要性

敏感性分析的目的:
- 了解每个因子对策略绩效的影响程度
- 识别关键因子和次要因子
- 辅助因子筛选和权重调整

分析方法 - 权重扫描法:
对于每个因子:
1. 保持其他因子权重不变
2. 将该因子权重从0逐步调整到0.5
3. 观察Calmar比率的变化
4. 计算敏感度 = Calmar的标准差
5. 敏感度越高，该因子越重要

因子重要性分类:
- 高敏感性 (>0.8): 因子轻微变化就显著影响绩效，需重点关注
- 中敏感性 (0.5-0.8): 因子有影响但不那么关键
- 低敏感性 (<0.5): 因子影响较小，可考虑剔除或降低权重

局限性:
- 只考虑单因子变化，忽略了因子间相互作用
- 依赖于最优权重的位置
- 在不同市场阶段敏感性可能不同
"""
import numpy as np
import pandas as pd
from typing import Dict


class SensitivityAnalyzer:
    """
    敏感性分析器 - 评估因子重要性
    """

    def __init__(self, evaluator):
        """
        Args:
            evaluator: FitnessEvaluator实例，用于评估不同权重配置
        """
        self.evaluator = evaluator

    def analyze_factor_importance(self,
                                  optimal_weights: Dict[str, float],
                                  data,
                                  factor_range: tuple = (0.0, 0.5),
                                  steps: int = 10) -> pd.DataFrame:
        """
        分析每个因子的重要性

        对每个因子:
        1. 在factor_range范围内取steps个点
        2. 固定其他因子为optimal_weights中的值
        3. 计算该因子每个取值对应的Calmar
        4. 计算Calmar序列的标准差作为敏感度

        公式:
        sensitivity = std(Calmar_values)

        Args:
            optimal_weights: GA找到的最优权重配置
            data: (start_date, end_date) 评估时间段
            factor_range: 测试的权重范围，如(0.0, 0.5)
            steps: 每个因子测试的点数

        Returns:
            DataFrame，列包括:
            - factor: 因子名称
            - base_weight: 最优权重
            - sensitivity: 敏感度(标准差)
            - fitness_range: Calmar的最大-最小值
            按sensitivity降序排列
        """
        results = []

        # 对每个因子单独测试
        for factor, base_weight in optimal_weights.items():
            sensitivities = []

            # 在权重范围内采样steps个点
            for pct in np.linspace(factor_range[0], factor_range[1], steps):
                # 构建测试权重
                test_weights = optimal_weights.copy()
                test_weights[factor] = pct

                # 归一化使和为1
                total = sum(test_weights.values())
                test_weights = {k: v / total for k, v in test_weights.items()}

                # 评估Calmar
                fitness = self.evaluator.evaluate(test_weights, data[0], data[1])
                sensitivities.append(fitness)

            # 计算敏感度: 标准差
            sensitivity = np.std(sensitivities)

            results.append({
                'factor': factor,           # 因子名
                'base_weight': base_weight, # 最优权重
                'sensitivity': sensitivity,  # 敏感度
                'fitness_range': max(sensitivities) - min(sensitivities)  # 收益范围
            })

        # 按敏感度降序排列
        return pd.DataFrame(results).sort_values('sensitivity', ascending=False)

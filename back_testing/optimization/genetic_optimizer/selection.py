"""
Selection: 选择算子 - 锦标赛选择法

选择操作从当前种群中选择个体作为父代，参与交叉产生子代。
选择策略影响GA的收敛速度和全局搜索能力。

锦标赛选择(Tournament Selection)是一种常用的选择方法:
- 每次从种群中随机抽取k个个体
- 让这k个个体竞争(比较适应度)
- 适应度最高者获胜，被选为父代
- 重复此过程直到选够所需父代数量

优点:
- 计算简单高效，适合并行化
- 自适应的选择压力: 个体差异大时选择压力大，差异小时压力小
- 避免过早收敛到局部最优
"""
import numpy as np
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    """
    锦标赛选择: 从k个随机个体中选择最优者

    流程:
    1. 从种群中随机不放回抽取k个个体
    2. 比较这k个个体的适应度
    3. 返回适应度最高的个体(深拷贝)

    Args:
        population: 当前种群
        k: 锦标赛规模，通常取2-5
           - k越大: 选择压力越大，优生优育
           - k越小: 种群多样性保持更好

    Returns:
        被选中的最优染色体(副本)
    """
    # 随机抽取k个不重复的个体索引
    tournament_idx = np.random.choice(len(population), size=k, replace=False)

    # 构建锦标赛: (个体索引, 适应度) 列表
    # 适应度为None的个体视为负无穷，不可能获胜
    tournament = [
        (i, population[i].fitness if population[i].fitness is not None else float('-inf'))
        for i in tournament_idx
    ]

    # 选择适应度最高的个体
    winner_idx = max(tournament, key=lambda x: x[1])[0]

    # 返回获胜者的副本(避免引用问题)
    return population[winner_idx].copy()


def select_parents(population: List[Chromosome], n_parents: int, k: int = 3) -> List[Chromosome]:
    """
    选择多个父代用于交叉

    通过多次锦标赛选择产生n_parents个父代。
    每次选择都是独立的，使用相同的锦标赛规模k。

    Args:
        population: 当前种群
        n_parents: 需要选择的父代数量
        k: 锦标赛规模

    Returns:
        选中的父代染色体列表
    """
    return [tournament_selection(population, k) for _ in range(n_parents)]

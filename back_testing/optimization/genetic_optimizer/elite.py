"""
Elite: 精英保留策略

精英保留(Elitism)是遗传算法中的一种策略:
- 每一代结束后，直接将最优的几个个体复制到下一代
- 不经过选择、交叉、变异操作
- 确保已找到的优秀解不会被破坏

作用:
- 加速收敛: 优秀基因能快速扩散
- 防止倒退: 最优解的适应度不会下降
- 安全性: 保留探索过程中发现的好解

注意事项:
- 精英比例不宜过高(通常5%-15%)
- 过高会导致种群多样性下降，易陷入局部最优
- 精英个体也会参与选择作为父代
"""
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def preserve_elite(population: List[Chromosome],
                   elite_ratio: float = 0.1) -> List[Chromosome]:
    """
    选择精英个体

    将种群按适应度降序排序，返回前elite_ratio比例的个体。

    流程:
    1. 计算精英数量: max(1, pop_size * elite_ratio)
       至少保留1个精英，即使比例很低
    2. 按适应度降序排序
    3. 返回前n_elite个精英的深拷贝

    Args:
        population: 当前种群
        elite_ratio: 精英比例，如0.1表示前10%

    Returns:
        精英染色体列表(副本)
    """
    # 计算精英数量，至少保留1个
    n_elite = max(1, int(len(population) * elite_ratio))

    # 按适应度降序排序
    # 适应度为None视为负无穷，排在最后
    sorted_pop = sorted(
        population,
        key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
        reverse=True
    )

    # 返回精英个体的副本
    return [c.copy() for c in sorted_pop[:n_elite]]

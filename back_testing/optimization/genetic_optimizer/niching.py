"""
Niching: 适应度共享 - 维持种群多样性

Niching(小生境)技术:
- 模仿自然生态中的"小生境"概念
- 同一生态位中的物种会竞争，限制单一物种过度繁殖
- 在GA中，防止相似个体过多，维持种群多样性

问题:
- 标准GA容易"早熟收敛"
- 种群过早聚集在某个局部最优解附近
- 丢失了探索其他区域的潜力

解决方案 - 适应度共享(Fitness Sharing):
- 计算每个个体周围有多少"邻居"(相似个体)
- 邻居越多，该个体的适应度打折越多
- 这样相似的个体之间产生竞争
- 鼓励不同类型的个体共存

参数:
- sigma(共享半径): 判定两个体是否"相似"的距离阈值
  - 常用值: 0.1 (基于权重向量的欧氏距离)
  - 距离 < sigma 认为处于同一生态位
- alpha(共享强度): 适应度折扣系数
  - 公式: sharing = 1 - (distance/sigma)^alpha
  - alpha=1时为线性共享
"""
import numpy as np
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def calculate_distance(chrom1: Chromosome, chrom2: Chromosome) -> float:
    """
    计算两个染色体之间的欧氏距离

    欧氏距离 = sqrt(sum((gene1_i - gene2_i)^2))

    用于判断两个个体是否"相似":
    - 距离小: 个体相似度高
    - 距离大: 个体差异大

    Args:
        chrom1: 第一个染色体
        chrom2: 第二个染色体

    Returns:
        欧氏距离
    """
    return np.sqrt(np.sum((chrom1.genes - chrom2.genes) ** 2))


def apply_niching(population: List[Chromosome],
                  sigma: float = 0.1,
                  alpha: float = 1.0) -> List[Chromosome]:
    """
    对种群应用适应度共享

    算法:
    对于每个个体i:
    1. 统计与i距离 < sigma的个体数量(小生境计数)
    2. 计算共享系数sharing
    3. 调整后的适应度 = 原适应度 / niche_count

    共享系数计算:
    sharing = 1 - (distance/sigma)^alpha  当 distance < sigma
    sharing = 0  当 distance >= sigma

    这样，处于拥挤区域的个体会被惩罚，
    稀有个体保持较高的相对适应度。

    Args:
        population: 当前种群
        sigma: 共享半径，超过此距离认为不在同一生态位
        alpha: 共享强度系数

    Returns:
        适应度调整后的种群(原地修改)
    """
    for i, chrom_i in enumerate(population):
        if chrom_i.fitness is None:
            continue

        niche_count = 0.0

        # 统计邻居数量
        for j, chrom_j in enumerate(population):
            if i == j or chrom_j.fitness is None:
                continue

            distance = calculate_distance(chrom_i, chrom_j)

            if distance < sigma:
                # 在共享半径内，计算sharing系数并累加
                sharing = 1.0 - (distance / sigma) ** alpha
                niche_count += sharing

        # 如果有邻居，应用适应度折扣
        if niche_count > 0:
            population[i].fitness = chrom_i.fitness / niche_count

    return population

"""
Crossover: 交叉算子 - 模拟二进制交叉(SBX)

交叉操作模拟生物的有性繁殖:
- 两个父代个体(染色体)交换部分基因
- 产生两个子代个体
- 子代继承父代的部分特征

SBX(Simulated Binary Crossover)是一种专门针对实数编码的交叉算子:
- 模仿二进制交叉的思想，但操作实数向量
- eta参数控制子代与父代的接近程度:
  - eta大(如20): 子代更接近父代，探索精细
  - eta小(如2): 子代更远离父代，探索激进
- 通过beta系数混合两个父代的基因

参数:
- crossover_rate: 交叉概率，通常0.6-0.9
  - 每对父代有crossover_rate的概率进行交叉
  - 否则直接复制父代为子代
"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def simulated_binary_crossover(parent1: Chromosome, parent2: Chromosome,
                               eta: float = 20.0) -> tuple:
    """
    模拟二进制交叉(SBX)

    对于每一对基因(两个父代的同一个位置):
    1. 随机决定是否交换这对基因(50%概率)
    2. 如果交换，根据eta计算混合系数beta
    3. 使用beta混合两个父代的基因值

    SBX公式:
    - u ~ Uniform(0,1)
    - beta = (2u)^(1/(eta+1))  当 u <= 0.5
    - beta = (1/(2(1-u)))^(1/(eta+1))  当 u > 0.5

    然后:
    - child1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
    - child2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)

    Args:
        parent1: 第一个父代染色体
        parent2: 第二个父代染色体
        eta: 分布指数，控制子代与父代的距离
             eta越大，子代越接近父代(探索精细)
             eta越小，子代越远离父代(探索激进)

    Returns:
        (child1, child2): 两个子代染色体
    """
    n_genes = len(parent1.genes)
    child1_genes = np.zeros(n_genes)
    child2_genes = np.zeros(n_genes)

    for i in range(n_genes):
        # 每个基因位独立决定是否交叉
        if np.random.rand() < 0.5:
            # 需要交叉，计算beta系数
            if abs(parent1.genes[i] - parent2.genes[i]) > 1e-10:
                # 父代基因值不同，可以进行有效交叉
                u = np.random.rand()

                # 根据u的大小选择beta公式
                # 确保beta >= 0
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

                # 使用beta混合两个父代的基因
                child1_genes[i] = 0.5 * ((1 + beta) * parent1.genes[i] +
                                          (1 - beta) * parent2.genes[i])
                child2_genes[i] = 0.5 * ((1 - beta) * parent1.genes[i] +
                                          (1 + beta) * parent2.genes[i])
            else:
                # 父代基因相同，交叉无意义，子代直接复制父代
                child1_genes[i] = parent1.genes[i]
                child2_genes[i] = parent2.genes[i]
        else:
            # 不交叉，子代直接复制对应父代的基因
            child1_genes[i] = parent1.genes[i]
            child2_genes[i] = parent2.genes[i]

    # 创建子代染色体对象
    child1 = Chromosome(genes=child1_genes)
    child2 = Chromosome(genes=child2_genes)

    return child1, child2


def crossover(parent1: Chromosome, parent2: Chromosome,
             crossover_rate: float = 0.7) -> tuple:
    """
    带概率的交叉操作

    以crossover_rate的概率执行SBX交叉，
    否则直接复制父代为子代(保持种群数量稳定)。

    Args:
        parent1: 第一个父代
        parent2: 第二个父代
        crossover_rate: 交叉概率，通常0.6-0.9
                        0.7表示70%概率执行交叉

    Returns:
        (child1, child2): 两个子代染色体
    """
    if np.random.rand() < crossover_rate:
        # 执行SBX交叉
        return simulated_binary_crossover(parent1, parent2)
    else:
        # 不交叉，直接复制父代
        return parent1.copy(), parent2.copy()

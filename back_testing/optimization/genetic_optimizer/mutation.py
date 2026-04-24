"""
Mutation: 变异算子 - 高斯变异

变异操作模拟生物的基因突变:
- 随机改变个体的某些基因
- 引入新的遗传信息，增加种群多样性
- 防止算法过早收敛到局部最优

高斯变异(Gaussian Mutation):
- 在选中变异的基因上添加均值为0的高斯噪声(正态分布)
- 噪声标准差(noise参数)控制变异幅度
- 变异后重新归一化确保权重和=1.0

参数:
- mutation_rate: 变异率，通常0.01-0.1
  - 指每个基因变异的概率，不是整个个体
  - 0.05表示每个基因有5%的概率变异
  - 变异率过高会破坏优秀解，过低会降低探索能力

- noise: 高斯噪声标准差，控制变异幅度
  - 0.1表示在原权重基础上加减约0.1的扰动
  - 噪声过大会破坏好解，过小则搜索缓慢
"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def gaussian_mutation(chromosome: Chromosome,
                     mutation_rate: float = 0.05,
                     noise: float = 0.1) -> Chromosome:
    """
    高斯变异

    对染色体的每个基因:
    1. 生成随机数决定是否变异(根据mutation_rate)
    2. 如果变异，在该基因上添加高斯噪声
    3. 将变异后的基因裁剪到[MIN_WEIGHT, MAX_WEIGHT]
    4. 所有基因变异后，归一化使总和=1.0

    注意: 变异是针对每个基因独立判断的
         种群大小50，基因数10，mutation_rate=0.05
         意味着平均每代有50*10*0.05=25个基因发生变异

    Args:
        chromosome: 要变异的染色体
        mutation_rate: 每个基因变异的概率(0-1)
        noise: 高斯噪声的标准差

    Returns:
        变异后的新染色体(副本)
    """
    mutated = chromosome.copy()
    n_genes = len(mutated.genes)

    for i in range(n_genes):
        # 根据mutation_rate决定是否变异此基因
        if np.random.rand() < mutation_rate:
            # 添加高斯噪声: 均值0，标准差=noise
            mutated.genes[i] += np.random.normal(0, noise)

            # 裁剪到合法范围
            mutated.genes[i] = np.clip(
                mutated.genes[i],
                Chromosome.MIN_WEIGHT,
                Chromosome.MAX_WEIGHT
            )

    # 归一化: 确保所有权重之和=1.0
    # 这里直接除以总和，因为前面已经裁剪到正数范围
    mutated.genes = mutated.genes / mutated.genes.sum()

    return mutated


def mutate_population(population: list,
                     mutation_rate: float = 0.05,
                     noise: float = 0.1) -> list:
    """
    对整个种群应用变异

    对种群中的每个个体调用高斯变异。
    注意: 这不会修改原种群，而是返回新的变异后个体列表。

    Args:
        population: 个体列表
        mutation_rate: 变异率
        noise: 高斯噪声标准差

    Returns:
        变异后的新个体列表
    """
    return [gaussian_mutation(c, mutation_rate, noise) for c in population]

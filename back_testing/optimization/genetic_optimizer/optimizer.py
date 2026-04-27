"""
Optimizer: 遗传算法主循环

遗传算法(GA)是一种模拟自然选择的优化算法:
- 初始化: 创建随机种群
- 评估: 计算每个个体的适应度
- 选择: 选拔优秀个体作为父代
- 交叉: 父代繁殖产生子代
- 变异: 子代基因随机改变
- 迭代: 重复评估-选择-繁殖，直到收敛或达到最大代数

核心思想:
- 适应度高的个体更可能被选中繁殖
- 交叉让好解的特征组合在一起
- 变异引入新的可能性
- 精英保留确保好解不会丢失

收敛判断:
- 达到最大代数
- 早停(patience): 验证集上连续N代无提升则停止
"""
import numpy as np
import time
from typing import Dict, Callable, Optional
from back_testing.optimization.genetic_optimizer.population import Population
from back_testing.optimization.genetic_optimizer.selection import tournament_selection
from back_testing.optimization.genetic_optimizer.crossover import crossover
from back_testing.optimization.genetic_optimizer.mutation import gaussian_mutation
from back_testing.optimization.genetic_optimizer.elite import preserve_elite


class GeneticOptimizer:
    """
    遗传算法优化器

    用于优化因子权重配置，最大化Calmar比率。

    GA流程:
    1. 初始化种群(population_size个随机染色体)
    2. 评估所有个体的适应度
    3. 保留精英个体(elite)
    4. 生成新个体:
       - 锦标赛选择父代
       - 交叉产生子代
       - 高斯变异
    5. 替换旧种群
    6. 重复3-5直到满足终止条件
    """

    def __init__(self,
                 population_size: int = 50,
                 max_generations: int = 100,
                 elite_ratio: float = 0.1,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.05,
                 tournament_k: int = 3,
                 patience: int = 20,
                 seed: int = None):
        """
        初始化GA优化器

        Args:
            population_size: 种群规模
                            通常设为参数数量的5倍
                            50 = 10个因子 × 5
            max_generations: 最大迭代代数
                            100代通常足够收敛
            elite_ratio: 精英保留比例
                        0.1 = 前10%直接进入下一代
            crossover_rate: 交叉概率
                           0.7 = 70%概率执行SBX交叉
            mutation_rate: 变异率(每基因)
                         0.05 = 每个基因5%概率变异
            tournament_k: 锦标赛规模
                         3 = 每次从3个个体中选1个
            patience: 早停耐心值
                    验证集上连续20代无提升则停止
            seed: 随机种子，用于复现结果
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.patience = patience

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)

    def optimize(self,
                fitness_func: Callable,
                train_data,
                val_data: Optional = None,
                val_fitness_func: Optional[Callable] = None,
                verbose: bool = True) -> Dict[str, float]:
        """
        执行遗传算法优化

        主循环:
        1. 初始化种群
        2. 评估初始种群适应度
        3. 保留精英
        4. 迭代:
           - 生成新个体(选择+交叉+变异)
           - 评估新个体适应度
           - 早停检查(如果有验证集)
           - 更新种群
        5. 返回最优个体

        Args:
            fitness_func: 适应度函数，签名: (weights_dict, train_data) -> float
                         返回IR/Calmar比率
            train_data: 训练数据，传递给fitness_func
            val_data: 验证数据，传递给val_fitness_func(可选)
            val_fitness_func: 验证适应度函数，签名同fitness_func
                             用于早停判断。为None时回退到训练集适应度(不推荐)
            verbose: 是否打印进度

        Returns:
            最优权重字典
        """
        eval_count = 0  # 评估次数计数
        start_time = time.time()
        gen_start_time = start_time

        # 步骤1: 初始化种群
        population = Population(size=self.population_size)
        if verbose:
            print(f"[初始化] 种群大小: {self.population_size}, 最大代数: {self.max_generations}")

        # 步骤2: 评估初始种群适应度
        for chrom in population.individuals:
            chrom.fitness = fitness_func(chrom.to_dict(), train_data)
            eval_count += 1

        if verbose:
            elapsed = time.time() - start_time
            best = population.get_best()
            ir_str = f"{best.fitness:.4f}" if best and best.fitness is not None else 'N/A'
            print(f"[代 0/{self.max_generations}] 初始种群完成 | 最优IR: {ir_str} | 耗时: {elapsed:.1f}s | 评估次数: {eval_count}")

        # 步骤3: 保留精英
        elite = preserve_elite(population.individuals, self.elite_ratio)

        # 早停相关
        no_improve_count = 0
        best_val_fitness = float('-inf')
        best_weights = None

        # 步骤4: 主循环
        for gen in range(self.max_generations):
            gen_start_time = time.time()

            new_individuals = []

            # 先放入精英个体(不经过选择交叉变异)
            new_individuals.extend(elite)

            # 生成新个体直到填满种群
            children_count = 0
            while len(new_individuals) < self.population_size:
                # 选择两个父代
                parent1 = tournament_selection(population.individuals, self.tournament_k)
                parent2 = tournament_selection(population.individuals, self.tournament_k)

                # 交叉
                child1, child2 = crossover(parent1, parent2, self.crossover_rate)

                # 变异
                child1 = gaussian_mutation(child1, self.mutation_rate)
                child2 = gaussian_mutation(child2, self.mutation_rate)

                # 评估子代适应度
                child1.fitness = fitness_func(child1.to_dict(), train_data)
                child2.fitness = fitness_func(child2.to_dict(), train_data)
                eval_count += 2
                children_count += 2

                new_individuals.extend([child1, child2])

            # 替换旧种群
            population.replace_individuals(new_individuals[:self.population_size])

            # 重新保留精英
            elite = preserve_elite(population.individuals, self.elite_ratio)

            # 当前代最优
            current_best = population.get_best()
            gen_elapsed = time.time() - gen_start_time
            total_elapsed = time.time() - start_time

            if verbose:
                ir_str = f"{current_best.fitness:.4f}" if current_best and current_best.fitness is not None else 'N/A'
                print(f"[代 {gen + 1}/{self.max_generations}] | 最优IR: {ir_str} | 代耗时: {gen_elapsed:.1f}s | 累计: {total_elapsed:.1f}s | 评估: {eval_count}")

            # 早停检查
            if val_data is not None:
                # 在验证集上评估当前最优（真正样本外评估）
                current_best_individual = population.get_best()
                if val_fitness_func is not None and current_best_individual is not None:
                    val_best = val_fitness_func(current_best_individual.to_dict(), val_data)
                else:
                    # 回退：使用训练集适应度（不推荐，但保持向后兼容）
                    val_best = max(
                        c.fitness for c in population.individuals
                        if c.fitness is not None
                    )

                if val_best > best_val_fitness:
                    # 有提升，更新最佳
                    improvement = val_best - best_val_fitness if best_val_fitness != float('-inf') else 0
                    best_val_fitness = val_best
                    no_improve_count = 0
                    best_weights = population.get_best().to_dict()
                    if verbose:
                        print(f"  ↳ [验证] IR提升: {val_best:.4f} (+{improvement:.4f}) ★ 新最优")
                else:
                    no_improve_count += 1
                    if verbose:
                        print(f"  ↳ [验证] 无提升 ({no_improve_count}/{self.patience})")

                # 早停判断
                if no_improve_count >= self.patience:
                    if verbose:
                        print(f"\n=== 早停: 验证集连续{self.patience}代无提升 ===")
                    break

        # 返回最优解
        if best_weights is None:
            best_weights = population.get_best().to_dict()

        if verbose:
            total_elapsed = time.time() - start_time
            print(f"\n=== 优化完成 ===")
            print(f"总评估次数: {eval_count} | 总耗时: {total_elapsed:.1f}s")
            print(f"实际迭代代数: {gen + 1}/{self.max_generations}")

        return best_weights

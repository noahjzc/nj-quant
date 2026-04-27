"""
Population: 种群管理 - 遗传算法的种群容器

种群(Population)是遗传算法的核心概念，包含多个染色体个体。
种群通过选择、交叉、变异操作逐代进化，最终产生最优解。

种群规模(pop_size)通常设置为待优化参数数量的5-10倍。
本项目中10个因子，种群大小设为50。
"""
from typing import List, Optional
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


class Population:
    """
    种群类 - 管理一群染色体

    提供种群的创建、排序、精英个体获取、个体替换等操作。
    每次迭代(generation)后，用新个体替换旧个体。
    """

    def __init__(self, size: int = 50):
        """
        创建种群

        Args:
            size: 种群规模(个体数量)
                - 太小: 搜索空间有限，易陷入局部最优
                - 太大: 计算量大，收敛慢
                - 经验值: 10参数 × 5 = 50
        """
        self.size = size
        # 个体列表
        self.individuals: List[Chromosome] = []
        # 初始化: 创建size个随机染色体
        self._initialize()

    def _initialize(self):
        """
        初始化种群

        创建size个随机初始染色体，形成第一代(generation 0)。
        随机初始化保证种群多样性，让GA有足够的搜索起点。
        """
        self.individuals = [Chromosome() for _ in range(self.size)]

    def sort_by_fitness(self, descending: bool = True):
        """
        按适应度排序

        Args:
            descending: True=降序(适应度高的在前)，False=升序
        """
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
            reverse=descending
        )

    def get_best(self) -> Optional[Chromosome]:
        """
        获取最优个体(适应度最高)

        用于:
        - 每代结束后获取当前最优解
        - 算法终止时返回最终解

        Returns:
            最优染色体，如果种群为空返回None
        """
        if not self.individuals:
            return None
        self.sort_by_fitness()
        return self.individuals[0]

    def get_elite(self, ratio: float = 0.1) -> List[Chromosome]:
        """
        获取精英个体

        精英保留(elitism)是GA的重要策略:
        - 直接将当前代最优的top%个体复制到下一代
        - 防止已找到的优秀解在选择中被淘汰

        Args:
            ratio: 精英比例，如0.1表示保留前10%

        Returns:
            精英染色体列表
        """
        n_elite = max(1, int(len(self.individuals) * ratio))
        self.sort_by_fitness()
        return [c.copy() for c in self.individuals[:n_elite]]

    def replace_individuals(self, new_individuals: List[Chromosome]):
        """
        用新个体替换当前种群

        替换策略:
        1. 如果新个体数量 >= 种群规模，直接截取前size个
        2. 如果新个体数量 < 种群规模:
           - 先放入所有新个体
           - 按适应度排序当前种群
           - 用当前最优个体填充剩余位置

        这种策略保证:
        - 精英个体有更高概率保留
        - 种群规模保持稳定

        Args:
            new_individuals: 新个体列表
        """
        if len(new_individuals) >= self.size:
            # 新个体足够，直接截取
            self.individuals = new_individuals[:self.size]
        else:
            # 需要填充: 保存当前个体用于补充
            current = list(self.individuals)
            self.individuals = list(new_individuals)

            # 计算需要补充的数量
            remaining = self.size - len(new_individuals)

            # 按适应度排序当前种群，保留最优的补充到新种群
            current.sort(
                key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
                reverse=True
            )
            # 补充最优个体
            self.individuals = list(new_individuals) + [c.copy() for c in current[:remaining]]

    def __len__(self):
        """返回种群规模"""
        return len(self.individuals)

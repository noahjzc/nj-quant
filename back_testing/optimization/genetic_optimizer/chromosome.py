"""
Chromosome: 染色体 - 实数向量编码因子权重

遗传算法中的个体，代表一套因子权重配置。
每个染色体包含多个基因，每个基因对应一个因子的权重。

因子顺序: RSI_1, RSI_2, RSI_3, KDJ_K, KDJ_D, MA_5, MA_10, MA_20, MA_30

约束条件:
- 单因子权重范围: [0.01, 0.40] (防止某因子权重过大或被完全排除)
- 权重之和 = 1.0 (归一化，保证可解释性)

示例染色体 (9个基因):
    [0.20, 0.10, 0.05, 0.15, 0.05, 0.15, 0.10, 0.10, 0.10]
     (RSI_1, RSI_2, RSI_3, KDJ_K, KDJ_D, MA_5, MA_10, MA_20, MA_30)
"""
import numpy as np
from typing import Dict


class Chromosome:
    """
    染色体类 - 编码因子权重

    遗传算法中的个体(individual)，对应一套因子权重配置。
    通过选择、交叉、变异操作进化，寻找最优权重组合。
    """

    # 因子名称列表，按固定顺序排列
    # 与 factor_config.py 保持一致，共12个因子
    #   RSI: RSI_1, RSI_2, RSI_3 (数值越高越好)
    #   KDJ: KDJ_K, KDJ_D (数值越高越好)
    #   MA: MA_5, MA_10, MA_20, MA_30 (多头排列越好)
    #   Momentum: RET_20, RET_60 (动量，越强越好)
    #   Size: LN_MCAP (对数市值，越小越好)
    FACTOR_NAMES = [
        'RSI_1', 'RSI_2', 'RSI_3', 'KDJ_K', 'KDJ_D',
        'MA_5', 'MA_10', 'MA_20', 'MA_30',
        'RET_20', 'RET_60', 'LN_MCAP'
    ]

    # 单因子权重下限: 0.01
    # 防止因子被完全排除在组合之外
    MIN_WEIGHT = 0.01

    # 单因子权重上限: 0.40
    # 防止单一因子主导整个组合，降低过拟合风险
    MAX_WEIGHT = 0.40

    def __init__(self, weights: Dict[str, float] = None, genes: np.ndarray = None):
        """
        创建染色体实例

        Args:
            weights: 因子权重字典，格式 {'因子名': 权重值}
                    例如: {'PB': 0.15, 'PE_TTM': 0.10, ...}
            genes: 直接传入基因数组(用于GA内部操作)
                   通常从weights转换而来，或通过交叉/变异产生
        """
        if weights is not None:
            # 从权重字典初始化: 转换为基因数组并归一化
            self.genes = self._weights_to_genes(weights)
        elif genes is not None:
            # 直接从基因数组初始化，复制一份避免共享引用
            self.genes = genes.copy()
        else:
            # 随机初始化: 生成均匀分布的随机数，然后归一化
            self.genes = self._random_init()

        # 适应度值: 初始为None，在评估后更新
        # Calmar比率 = 年化收益率 / 最大回撤
        self.fitness = None

    def _weights_to_genes(self, weights: Dict[str, float]) -> np.ndarray:
        """
        将权重字典转换为基因数组并归一化

        Args:
            weights: 权重字典

        Returns:
            归一化后的基因数组，元素之和=1.0
        """
        # 按FACTOR_NAMES顺序提取权重，缺失默认为0
        genes = np.array([weights.get(f, 0.0) for f in self.FACTOR_NAMES])
        return self._normalize(genes)

    def _genes_to_weights(self) -> Dict[str, float]:
        """
        将基因数组转换为权重字典

        Returns:
            权重字典，格式 {'因子名': 权重值}
        """
        return {f: w for f, w in zip(self.FACTOR_NAMES, self.genes)}

    @staticmethod
    def _normalize(genes: np.ndarray) -> np.ndarray:
        """
        归一化基因数组: 确保和为1.0且每个基因在合法范围内

        步骤:
        1. 先归一化使总和=1.0
        2. 如果有基因超出[MIN_WEIGHT, MAX_WEIGHT]范围，裁剪并重新归一化
        3. 迭代直到所有基因都在范围内

        Args:
            genes: 原始基因数组

        Returns:
            归一化后的基因数组
        """
        for _ in range(10):  # 最多迭代10次，通常1-2次就收敛
            total = genes.sum()
            if total > 0:
                genes = genes / total
            else:
                genes = np.ones(len(genes)) / len(genes)

            # 检查是否有基因超出范围
            if np.all((genes >= Chromosome.MIN_WEIGHT) & (genes <= Chromosome.MAX_WEIGHT)):
                break

            # 裁剪到范围内
            genes = np.clip(genes, Chromosome.MIN_WEIGHT, Chromosome.MAX_WEIGHT)
        else:
            # 理论上不会走到这里，但以防万一
            genes = np.clip(genes, Chromosome.MIN_WEIGHT, Chromosome.MAX_WEIGHT)
            genes = genes / genes.sum() if genes.sum() > 0 else np.ones(len(genes)) / len(genes)

        return genes

    def _random_init(self) -> np.ndarray:
        """
        随机初始化基因数组

        使用均匀分布[0,1]生成初始基因，然后归一化。
        初始化的基因分布较为均匀，给GA足够的搜索空间。

        Returns:
            随机基因数组
        """
        genes = np.random.rand(len(self.FACTOR_NAMES))
        return self._normalize(genes)

    def to_dict(self) -> Dict[str, float]:
        """
        转换为权重字典

        用于输出和传递给其他组件(如回测器)。

        Returns:
            权重字典
        """
        return self._genes_to_weights()

    def copy(self) -> 'Chromosome':
        """
        深拷贝染色体

        复制基因数组和适应度值，用于:
        - 选择操作中返回个体副本
        - 交叉操作中创建子代

        Returns:
            新的染色体实例
        """
        new_chrom = Chromosome(genes=self.genes)
        new_chrom.fitness = self.fitness
        return new_chrom

    def mutate(self, gene_index: int, noise: float = 0.1):
        """
        对单个基因执行高斯变异

        变异操作:
        1. 在指定基因上添加均值0、标准差=noise的高斯噪声
        2. 重新归一化确保权重和=1.0

        Args:
            gene_index: 要变异的基因索引(0-9)
            noise: 高斯噪声的标准差，控制变异幅度
        """
        self.genes[gene_index] += np.random.normal(0, noise)
        self.genes = self._normalize(self.genes)

    def __repr__(self):
        """
        打印友好表示

        显示每个因子及其权重，权重保留3位小数
        """
        w = self.to_dict()
        return f"Chromosome({', '.join(f'{k}:{v:.3f}' for k, v in w.items())})"

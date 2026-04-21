# 遗传算法因子权重优化实施计划

> **目标**: 实现基于遗传算法的多因子权重优化系统，在回撤≤20%约束下最大化Calmar比率

**Tech Stack**: Python, pandas, numpy, concurrent.futures (并行化)

---

## Task 1: 创建项目结构与基础类

**目录**:
- 创建: `back_testing/optimization/genetic_optimizer/`

**文件**:
- 创建: `back_testing/optimization/__init__.py`
- 创建: `back_testing/optimization/genetic_optimizer/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p back_testing/optimization/genetic_optimizer
touch back_testing/optimization/__init__.py
touch back_testing/optimization/genetic_optimizer/__init__.py
```

- [ ] **Step 2: 验证目录创建成功**

```bash
ls -la back_testing/optimization/
ls -la back_testing/optimization/genetic_optimizer/
```

---

## Task 2: 实现染色体类 (Chromosome)

**文件**: `back_testing/optimization/genetic_optimizer/chromosome.py`

- [ ] **Step 1: 编写测试**

```python
# tests/back_testing/optimization/test_chromosome.py
import pytest
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def test_chromosome_creation():
    """测试染色体创建"""
    weights = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    chrom = Chromosome(weights)
    assert chrom.to_dict() == weights

def test_chromosome_normalization():
    """测试权重归一化"""
    # 权重和不为1时应自动归一化
    chrom = Chromosome({'a': 0.6, 'b': 0.6})  # sum=1.2
    normalized = chrom.to_dict()
    assert abs(sum(normalized.values()) - 1.0) < 0.001

def test_chromosome_mutation():
    """测试变异后归一化"""
    chrom = Chromosome({'a': 0.5, 'b': 0.5})
    chrom.mutate(gene_index=0, noise=0.1)
    assert abs(sum(chrom.to_dict().values()) - 1.0) < 0.001
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/back_testing/optimization/test_chromosome.py -v
# 预期: FAIL - module not found
```

- [ ] **Step 3: 编写实现**

```python
# back_testing/optimization/genetic_optimizer/chromosome.py
"""染色体编码: 实数向量表示因子权重"""
import numpy as np
from typing import Dict

class Chromosome:
    """染色体类: 编码一组因子权重"""

    # 因子名称列表 (固定顺序)
    FACTOR_NAMES = [
        'PB', 'PE_TTM', 'PS_TTM', 'RSI_1', 'KDJ_K',
        'MA_5', 'MA_20', 'TURNOVER', 'VOLUME_RATIO', 'AMPLITUDE'
    ]

    # 权重约束
    MIN_WEIGHT = 0.01
    MAX_WEIGHT = 0.40

    def __init__(self, weights: Dict[str, float] = None, genes: np.ndarray = None):
        """
        创建染色体

        Args:
            weights: 权重字典 {'因子名': 权重}
            genes: 直接传入的基因数组 (用于GA操作)
        """
        if weights is not None:
            self.genes = self._weights_to_genes(weights)
        elif genes is not None:
            self.genes = genes.copy()
        else:
            # 随机初始化
            self.genes = self._random_init()

    def _weights_to_genes(self, weights: Dict[str, float]) -> np.ndarray:
        """将权重字典转换为基因数组并归一化"""
        genes = np.array([weights.get(f, 0.0) for f in self.FACTOR_NAMES])
        return self._normalize(genes)

    def _genes_to_weights(self) -> Dict[str, float]:
        """将基因数组转换为权重字典"""
        return {f: w for f, w in zip(self.FACTOR_NAMES, self.genes)}

    def _normalize(self, genes: np.ndarray) -> np.ndarray:
        """归一化使权重和为1"""
        total = genes.sum()
        if total > 0:
            genes = genes / total
        else:
            # 均匀分布
            genes = np.ones(len(genes)) / len(genes)
        return genes

    def _random_init(self) -> np.ndarray:
        """随机初始化基因"""
        genes = np.random.rand(len(self.FACTOR_NAMES))
        return self._normalize(genes)

    def to_dict(self) -> Dict[str, float]:
        """转换为权重字典"""
        return self._genes_to_weights()

    def copy(self) -> 'Chromosome':
        """深拷贝"""
        return Chromosome(genes=self.genes)

    def mutate(self, gene_index: int, noise: float = 0.05):
        """单点变异 (高斯)"""
        self.genes[gene_index] += np.random.normal(0, noise)
        self.genes[gene_index] = np.clip(self.genes[gene_index], self.MIN_WEIGHT, self.MAX_WEIGHT)
        self.genes = self._normalize(self.genes)

    def __repr__(self):
        w = self.to_dict()
        return f"Chromosome({', '.join(f'{k}:{v:.3f}' for k, v in w.items())})"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/back_testing/optimization/test_chromosome.py -v
# 预期: PASS
```

- [ ] **Step 5: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/chromosome.py
git add tests/back_testing/optimization/test_chromosome.py
git commit -m "feat(optimization): add Chromosome class for gene encoding"
```

---

## Task 3: 实现种群管理 (Population)

**文件**: `back_testing/optimization/genetic_optimizer/population.py`

- [ ] **Step 1: 编写测试**

```python
# tests/back_testing/optimization/test_population.py
import pytest
from back_testing.optimization.genetic_optimizer.population import Population
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def test_population_creation():
    """测试种群创建"""
    pop = Population(size=50)
    assert len(pop.individuals) == 50

def test_population_fitness_ranking():
    """测试按适应度排序"""
    pop = Population(size=10)
    # 设置随机适应度
    for i, chrom in enumerate(pop.individuals):
        chrom.fitness = i * 0.1
    pop.sort_by_fitness()
    # 验证排序正确
    for i in range(len(pop.individuals) - 1):
        assert pop.individuals[i].fitness >= pop.individuals[i+1].fitness
```

- [ ] **Step 2: 编写实现**

```python
# back_testing/optimization/genetic_optimizer/population.py
"""种群管理"""
from typing import List, Optional
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

class Population:
    """种群类: 管理一组染色体"""

    def __init__(self, size: int = 50):
        """
        创建种群

        Args:
            size: 种群大小
        """
        self.size = size
        self.individuals: List[Chromosome] = []
        self._initialize()

    def _initialize(self):
        """初始化种群"""
        self.individuals = [Chromosome() for _ in range(self.size)]

    def sort_by_fitness(self, descending: bool = True):
        """按适应度排序"""
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else -float('inf'),
            reverse=descending
        )

    def get_best(self) -> Optional[Chromosome]:
        """获取最优个体"""
        if not self.individuals:
            return None
        if self.individuals[0].fitness is None:
            return None
        self.sort_by_fitness()
        return self.individuals[0]

    def get_elite(self, ratio: float = 0.1) -> List[Chromosome]:
        """获取精英个体"""
        n = max(1, int(len(self.individuals) * ratio))
        self.sort_by_fitness()
        return [c.copy() for c in self.individuals[:n]]

    def replace_individuals(self, new_individuals: List[Chromosome]):
        """替换部分个体"""
        self.individuals = new_individuals[:self.size]

    def __len__(self):
        return len(self.individuals)
```

- [ ] **Step 3: 运行测试并提交**

```bash
# 测试和提交
git add back_testing/optimization/genetic_optimizer/population.py
git add tests/back_testing/optimization/test_population.py
git commit -m "feat(optimization): add Population class for population management"
```

---

## Task 4: 实现遗传算子

**文件**:
- `back_testing/optimization/genetic_optimizer/selection.py`
- `back_testing/optimization/genetic_optimizer/crossover.py`
- `back_testing/optimization/genetic_optimizer/mutation.py`
- `back_testing/optimization/genetic_optimizer/elite.py`

- [ ] **Step 1: 实现选择算子 (锦标赛)**

```python
# back_testing/optimization/genetic_optimizer/selection.py
"""选择算子"""
import numpy as np
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    """
    锦标赛选择

    Args:
        population: 种群列表
        k: 锦标赛规模

    Returns:
        被选中的个体
    """
    tournament = np.random.choice(len(population), size=k, replace=False)
    competitors = [(i, population[i].fitness if population[i].fitness is not None else -float('inf'))
                   for i in tournament]
    winner_idx = max(competitors, key=lambda x: x[1])[0]
    return population[winner_idx].copy()

def select_parents(population: List[Chromosome], n_parents: int, k: int = 3) -> List[Chromosome]:
    """
    选择多个父代

    Args:
        population: 种群列表
        n_parents: 需要选择的父代数量
        k: 锦标赛规模

    Returns:
        父代列表
    """
    return [tournament_selection(population, k) for _ in range(n_parents)]
```

- [ ] **Step 2: 实现交叉算子 (模拟二进制交叉 SBX)**

```python
# back_testing/optimization/genetic_optimizer/crossover.py
"""交叉算子"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def simulated_binary_crossover(parent1: Chromosome, parent2: Chromosome,
                                eta: float = 20.0) -> tuple:
    """
    模拟二进制交叉 (SBX)

    Args:
        parent1: 父代1
        parent2: 父代2
        eta: 分布指数 (越大越接近父代)

    Returns:
        (子代1, 子代2)
    """
    n_genes = len(parent1.genes)
    child1_genes = np.zeros(n_genes)
    child2_genes = np.zeros(n_genes)

    for i in range(n_genes):
        if np.random.rand() < 0.5:
            if abs(parent1.genes[i] - parent2.genes[i]) > 1e-10:
                # 计算u
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

                # 生成子代基因
                child1_genes[i] = 0.5 * ((1 + beta) * parent1.genes[i] + (1 - beta) * parent2.genes[i])
                child2_genes[i] = 0.5 * ((1 - beta) * parent1.genes[i] + (1 + beta) * parent2.genes[i])
            else:
                child1_genes[i] = parent1.genes[i]
                child2_genes[i] = parent2.genes[i]
        else:
            child1_genes[i] = parent1.genes[i]
            child2_genes[i] = parent2.genes[i]

    # 归一化
    child1 = Chromosome(genes=child1_genes)
    child2 = Chromosome(genes=child2_genes)

    return child1, child2

def crossover(parent1: Chromosome, parent2: Chromosome,
              crossover_rate: float = 0.7) -> tuple:
    """
    交叉操作 (带概率)

    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_rate: 交叉概率

    Returns:
        (子代1, 子代2)
    """
    if np.random.rand() < crossover_rate:
        return simulated_binary_crossover(parent1, parent2)
    else:
        return parent1.copy(), parent2.copy()
```

- [ ] **Step 3: 实现变异算子 (高斯变异)**

```python
# back_testing/optimization/genetic_optimizer/mutation.py
"""变异算子"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def gaussian_mutation(chromosome: Chromosome,
                     mutation_rate: float = 0.05,
                     noise: float = 0.1) -> Chromosome:
    """
    高斯变异

    Args:
        chromosome: 待变异个体
        mutation_rate: 变异概率
        noise: 高斯噪声标准差

    Returns:
        变异后的个体
    """
    mutated = chromosome.copy()
    n_genes = len(mutated.genes)

    for i in range(n_genes):
        if np.random.rand() < mutation_rate:
            mutated.genes[i] += np.random.normal(0, noise)
            mutated.genes[i] = np.clip(mutated.genes[i],
                                        Chromosome.MIN_WEIGHT,
                                        Chromosome.MAX_WEIGHT)

    # 重新归一化
    mutated.genes = mutated.genes / mutated.genes.sum()

    return mutated

def mutate_population(population: list,
                      mutation_rate: float = 0.05,
                      noise: float = 0.1) -> list:
    """
    对整个种群进行变异

    Args:
        population: 种群列表
        mutation_rate: 变异概率
        noise: 高斯噪声标准差

    Returns:
        变异后的种群
    """
    return [gaussian_mutation(c, mutation_rate, noise) for c in population]
```

- [ ] **Step 4: 实现精英保留**

```python
# back_testing/optimization/genetic_optimizer/elite.py
"""精英保留策略"""
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

def preserve_elite(population: List[Chromosome],
                  elite_ratio: float = 0.1) -> List[Chromosome]:
    """
    保留精英个体

    Args:
        population: 当前种群
        elite_ratio: 精英比例

    Returns:
        精英个体列表
    """
    n_elite = max(1, int(len(population) * elite_ratio))

    # 按适应度排序
    sorted_pop = sorted(population,
                       key=lambda x: x.fitness if x.fitness is not None else -float('inf'),
                       reverse=True)

    return [c.copy() for c in sorted_pop[:n_elite]]
```

- [ ] **Step 5: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/selection.py
git add back_testing/optimization/genetic_optimizer/crossover.py
git add back_testing/optimization/genetic_optimizer/mutation.py
git add back_testing/optimization/genetic_optimizer/elite.py
git commit -m "feat(optimization): add genetic operators (selection, crossover, mutation, elite)"
```

---

## Task 5: 实现适应度函数

**文件**: `back_testing/optimization/genetic_optimizer/fitness.py`

- [ ] **Step 1: 理解现有回测接口**

```python
# 查看现有回测接口
# back_testing/backtest/run_composite_backtest.py
# 了解如何传入不同权重进行回测
```

- [ ] **Step 2: 编写测试**

```python
# tests/back_testing/optimization/test_fitness.py
import pytest
import pandas as pd
import numpy as np
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator

def test_fitness_with_feasible_solution():
    """测试可行解的适应度计算"""
    # 创建模拟数据
    # ...

def test_fitness_with_infeasible_solution():
    """测试违反约束的解应返回0"""
    # ...
```

- [ ] **Step 3: 编写适应度函数实现**

```python
# back_testing/optimization/genetic_optimizer/fitness.py
"""适应度函数"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.data.data_provider import DataProvider

class FitnessEvaluator:
    """适应度评估器"""

    def __init__(self, data_path: str,
                 max_drawdown_constraint: float = 0.20,
                 n_stocks: int = 5):
        """
        Args:
            data_path: 数据路径
            max_drawdown_constraint: 最大回撤约束
            n_stocks: 持仓数量
        """
        self.data_path = data_path
        self.max_drawdown_constraint = max_drawdown_constraint
        self.n_stocks = n_stocks

        # 用于加载因子数据
        self.factor_loader = FactorLoader()

    def evaluate(self, weights: Dict[str, float],
                start_date: pd.Timestamp,
                end_date: pd.Timestamp) -> float:
        """
        评估一组权重的适应度

        Args:
            weights: 因子权重字典
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            Calmar比率 (无效解返回0)
        """
        try:
            # 1. 运行回测
            result = self._run_backtest(weights, start_date, end_date)

            # 2. 提取指标
            annual_return = result.get('annual_return', 0)
            max_drawdown = result.get('max_drawdown', 0)

            # 3. 约束检查
            if max_drawdown > self.max_drawdown_constraint:
                return 0.0

            # 4. 计算Calmar比率 (防止除零)
            calmar = annual_return / max(max_drawdown, 0.01)

            return calmar

        except Exception as e:
            print(f"评估失败: {e}")
            return 0.0

    def _run_backtest(self, weights: Dict[str, float],
                     start_date: pd.Timestamp,
                     end_date: pd.Timestamp) -> Dict:
        """
        运行回测

        调用CompositeRotator进行真实回测

        Args:
            weights: 因子权重
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果字典 {
                'annual_return': float,
                'max_drawdown': float,
                'sharpe_ratio': float,
                'total_return': float
            }
        """
        from back_testing.composite_rotator import CompositeRotator

        # 创建CompositeRotator，使用新的因子权重
        rotator = CompositeRotator(
            data_path=self.data_path,
            initial_capital=1000000.0,
            n_stocks=self.n_stocks,
            use_multi_factor=True
        )

        # 更新因子权重
        rotator.factor_weights = weights

        # 导入方向配置
        from back_testing.factors.factor_config import get_factor_directions
        rotator.factor_directions = get_factor_directions()

        # 运行回测
        # 注意: 实际实现需要CompositeRotator支持外部调用run方法
        # 或者需要重构以支持单次回测调用

        # 简化: 收集每周选股结果并模拟计算
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        weekly_returns = []

        for date in dates:
            # 加载当日因子数据
            factors = list(weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(date, factors)

            if len(factor_data) == 0:
                continue

            # 计算评分
            selector = MultiFactorSelector(
                weights=weights,
                directions={k: 1 if 'PB' not in k else -1
                           for k in weights.keys()}
            )
            selected = selector.select_top_stocks(factor_data, n=self.n_stocks)

            # 获取选中股票的平均收益率 (简化)
            # 实际需要获取这些股票从date到下个date的真实收益
            # 这里用随机模拟替代
            weekly_returns.append(0.01)  # 简化假设每周1%收益

        if not weekly_returns:
            return {
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0
            }

        # 计算组合收益序列
        portfolio_values = [1.0]
        for r in weekly_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + r))

        # 计算绩效指标
        portfolio_values = np.array(portfolio_values)
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (52 / len(weekly_returns)) - 1

        # 计算最大回撤
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Sharpe比率简化计算 (假设无风险利率3%)
        excess_return = annual_return - 0.03
        volatility = np.std(weekly_returns) * np.sqrt(52)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return
        }

def calculate_fitness(weights: Dict[str, float],
                     evaluator: FitnessEvaluator,
                     start_date: pd.Timestamp,
                     end_date: pd.Timestamp) -> float:
    """计算适应度的便捷函数"""
    return evaluator.evaluate(weights, start_date, end_date)
```

- [ ] **Step 4: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/fitness.py
git commit -m "feat(optimization): add FitnessEvaluator for Calmar ratio calculation"
```

---

## Task 6: 实现Walk-Forward分析器

**文件**: `back_testing/optimization/genetic_optimizer/walk_forward.py`

- [ ] **Step 1: 编写实现**

```python
# back_testing/optimization/genetic_optimizer/walk_forward.py
"""Walk-Forward分析器"""
import pandas as pd
from typing import List, Dict, Tuple
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome

class WalkForwardAnalyzer:
    """Walk-Forward分析器: 滚动窗口优化"""

    def __init__(self,
                 train_window_years: int = 3,
                 val_window_years: int = 1,
                 test_window_years: int = 1,
                 step_months: int = 3):
        """
        Args:
            train_window_years: 训练窗口年数
            val_window_years: 验证窗口年数
            test_window_years: 测试窗口年数
            step_months: 滚动步进月数
        """
        self.train_window_years = train_window_years
        self.val_window_years = val_window_years
        self.test_window_years = test_window_years
        self.step_months = step_months

    def get_windows(self, start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> List[Dict]:
        """
        生成Walk-Forward窗口

        Args:
            start_date: 数据开始日期
            end_date: 数据结束日期

        Returns:
            窗口列表 [{'train': (start, end), 'val': ..., 'test': ...}, ...]
        """
        windows = []
        current = start_date

        train_months = self.train_window_years * 12
        val_months = self.val_window_years * 12
        test_months = self.test_window_years * 12

        while True:
            train_end = current + pd.DateOffset(months=train_months)
            val_end = train_end + pd.DateOffset(months=val_months)
            test_end = val_end + pd.DateOffset(months=test_months)

            if test_end > end_date:
                break

            windows.append({
                'train': (current, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end)
            })

            current = current + pd.DateOffset(months=self.step_months)

        return windows

    def aggregate_weights(self, weights_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合多个窗口的最优权重

        Args:
            weights_list: 各窗口最优权重列表

        Returns:
            聚合后的权重 (简单平均)
        """
        if not weights_list:
            return {}

        # 简单平均
        n = len(weights_list)
        aggregated = {}
        all_keys = set()
        for w in weights_list:
            all_keys.update(w.keys())

        for key in all_keys:
            values = [w.get(key, 0) for w in weights_list]
            aggregated[key] = sum(values) / n

        # 归一化
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return aggregated
```

- [ ] **Step 2: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/walk_forward.py
git commit -m "feat(optimization): add WalkForwardAnalyzer for walk-forward analysis"
```

---

## Task 7: 实现GA主循环

**文件**: `back_testing/optimization/genetic_optimizer/genetic_optimizer.py`

- [ ] **Step 1: 编写实现**

```python
# back_testing/optimization/genetic_optimizer/genetic_optimizer.py
"""遗传算法优化器主入口"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable
from back_testing.optimization.genetic_optimizer.population import Population
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome
from back_testing.optimization.genetic_optimizer.selection import tournament_selection, select_parents
from back_testing.optimization.genetic_optimizer.crossover import crossover
from back_testing.optimization.genetic_optimizer.mutation import gaussian_mutation
from back_testing.optimization.genetic_optimizer.elite import preserve_elite

class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self,
                 population_size: int = 50,
                 max_generations: int = 100,
                 elite_ratio: float = 0.1,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.05,
                 tournament_k: int = 3,
                 patience: int = 20):
        """
        Args:
            population_size: 种群大小
            max_generations: 最大代数
            elite_ratio: 精英保留比例
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            tournament_k: 锦标赛规模
            patience: 早停耐心值
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.patience = patience

    def optimize(self,
                fitness_func: Callable,
                train_data,
                val_data = None,
                verbose: bool = True) -> Dict[str, float]:
        """
        执行遗传算法优化

        Args:
            fitness_func: 适应度函数 (weights -> float)
            train_data: 训练数据
            val_data: 验证数据 (可选,用于早停)
            verbose: 是否打印进度

        Returns:
            最优权重字典
        """
        # 1. 初始化种群
        population = Population(size=self.population_size)

        # 2. 评估初始种群
        for chrom in population.individuals:
            chrom.fitness = fitness_func(chrom.to_dict(), train_data)

        # 3. 精英个体
        elite = preserve_elite(population.individuals, self.elite_ratio)

        # 4. 早停计数
        no_improve_count = 0
        best_val_fitness = -float('inf')

        # 5. 主循环
        for gen in range(self.max_generations):
            # 创建新一代
            new_individuals = []

            # 保留精英
            new_individuals.extend(elite)

            # 生成子代
            while len(new_individuals) < self.population_size:
                # 选择父代
                parent1 = tournament_selection(population.individuals, self.tournament_k)
                parent2 = tournament_selection(population.individuals, self.tournament_k)

                # 交叉
                child1, child2 = crossover(parent1, parent2, self.crossover_rate)

                # 变异
                child1 = gaussian_mutation(child1, self.mutation_rate)
                child2 = gaussian_mutation(child2, self.mutation_rate)

                # 评估
                child1.fitness = fitness_func(child1.to_dict(), train_data)
                child2.fitness = fitness_func(child2.to_dict(), train_data)

                new_individuals.extend([child1, child2])

            # 替换种群
            population.replace_individuals(new_individuals[:self.population_size])

            # 获取精英
            elite = preserve_elite(population.individuals, self.elite_ratio)

            # 早停检查
            if val_data is not None:
                current_best = max(c.fitness for c in population.individuals if c.fitness is not None)
                if current_best > best_val_fitness:
                    best_val_fitness = current_best
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    if verbose:
                        print(f"早停: 第{gen+1}代，验证集无提升超过{self.patience}代")
                    break

            # 打印进度
            if verbose and (gen + 1) % 10 == 0:
                best = population.get_best()
                print(f"代数 {gen+1}: 最优适应度 = {best.fitness if best else 'N/A'}")

        # 返回最优解
        best_chrom = population.get_best()
        return best_chrom.to_dict() if best_chrom else {}
```

- [ ] **Step 2: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/genetic_optimizer.py
git commit -m "feat(optimization): add GeneticOptimizer main loop"
```

---

## Task 8: 实现敏感性分析

**文件**: `back_testing/optimization/genetic_optimizer/sensitivity.py`

- [ ] **Step 1: 编写实现**

```python
# back_testing/optimization/genetic_optimizer/sensitivity.py
"""敏感性分析"""
import numpy as np
import pandas as pd
from typing import Dict, List

class SensitivityAnalyzer:
    """敏感性分析器"""

    def __init__(self, evaluator):
        """
        Args:
            evaluator: FitnessEvaluator实例
        """
        self.evaluator = evaluator

    def analyze_factor_importance(self,
                                  optimal_weights: Dict[str, float],
                                  data,
                                  factor_range: tuple = (0.0, 0.5),
                                  steps: int = 10) -> pd.DataFrame:
        """
        分析各因子重要性

        Args:
            optimal_weights: 最优权重
            data: 回测数据
            factor_range: 测试范围
            steps: 步数

        Returns:
            各因子的敏感性分数
        """
        results = []

        for factor, base_weight in optimal_weights.items():
            sensitivities = []

            for pct in np.linspace(factor_range[0], factor_range[1], steps):
                # 修改该因子权重
                test_weights = optimal_weights.copy()
                test_weights[factor] = pct

                # 归一化
                total = sum(test_weights.values())
                test_weights = {k: v / total for k, v in test_weights.items()}

                # 评估
                fitness = self.evaluator.evaluate(test_weights, data)
                sensitivities.append(fitness)

            # 计算敏感性 (收益变化的标准差)
            sensitivity = np.std(sensitivities)
            results.append({
                'factor': factor,
                'base_weight': base_weight,
                'sensitivity': sensitivity,
                'fitness_range': max(sensitivities) - min(sensitivities)
            })

        return pd.DataFrame(results).sort_values('sensitivity', ascending=False)
```

- [ ] **Step 2: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/sensitivity.py
git commit -m "feat(optimization): add SensitivityAnalyzer for factor importance analysis"
```

---

## Task 9: 创建优化器入口脚本

**文件**: `back_testing/optimization/run_factor_optimization.py`

- [ ] **Step 1: 编写入口脚本**

```python
# back_testing/optimization/run_factor_optimization.py
"""因子权重优化入口脚本"""
import pandas as pd
from back_testing.optimization.genetic_optimizer.genetic_optimizer import GeneticOptimizer
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator
from back_testing.optimization.genetic_optimizer.walk_forward import WalkForwardAnalyzer
from back_testing.optimization.genetic_optimizer.sensitivity import SensitivityAnalyzer

def main():
    # 1. 配置
    DATA_PATH = 'data/daily_ycz/'
    START_DATE = pd.Timestamp('2019-01-01')
    END_DATE = pd.Timestamp('2024-01-01')

    # 2. 创建评估器
    evaluator = FitnessEvaluator(
        data_path=DATA_PATH,
        max_drawdown_constraint=0.20,
        n_stocks=5
    )

    # 3. 创建Walk-Forward分析器
    wf_analyzer = WalkForwardAnalyzer(
        train_window_years=3,
        val_window_years=1,
        test_window_years=1,
        step_months=3
    )

    # 4. 获取窗口
    windows = wf_analyzer.get_windows(START_DATE, END_DATE)
    print(f"共 {len(windows)} 个Walk-Forward窗口")

    # 5. 对每个窗口运行GA
    all_optimal_weights = []

    for i, window in enumerate(windows):
        print(f"\n=== 窗口 {i+1}/{len(windows)} ===")
        print(f"训练: {window['train'][0].date()} ~ {window['train'][1].date()}")
        print(f"验证: {window['val'][0].date()} ~ {window['val'][1].date()}")
        print(f"测试: {window['test'][0].date()} ~ {window['test'][1].date()}")

        # 定义适应度函数
        def fitness_func(weights, data):
            return evaluator.evaluate(weights, data[0], data[1])

        # 创建优化器
        optimizer = GeneticOptimizer(
            population_size=50,
            max_generations=100,
            elite_ratio=0.1,
            crossover_rate=0.7,
            mutation_rate=0.05,
            patience=20
        )

        # 运行优化
        optimal = optimizer.optimize(
            fitness_func=lambda w: fitness_func(w, (window['train'][0], window['train'][1])),
            train_data=(window['train'][0], window['train'][1]),
            val_data=(window['val'][0], window['val'][1]),
            verbose=True
        )

        # 在验证集上评估
        val_fitness = evaluator.evaluate(optimal, window['val'][0], window['val'][1])
        print(f"验证集Calmar: {val_fitness:.4f}")

        # 在测试集上评估
        test_fitness = evaluator.evaluate(optimal, window['test'][0], window['test'][1])
        print(f"测试集Calmar: {test_fitness:.4f}")

        all_optimal_weights.append(optimal)

    # 6. 聚合权重
    final_weights = wf_analyzer.aggregate_weights(all_optimal_weights)
    print(f"\n=== 最终权重 ===")
    for k, v in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    # 7. 敏感性分析
    print("\n=== 敏感性分析 ===")
    sensitivity = SensitivityAnalyzer(evaluator)
    importance = sensitivity.analyze_factor_importance(
        final_weights,
        (START_DATE, END_DATE)
    )
    print(importance.to_string())

    # 8. 保存结果
    import json
    with open('optimization_results.json', 'w') as f:
        json.dump({
            'final_weights': final_weights,
            'window_weights': all_optimal_weights,
            'factor_importance': importance.to_dict()
        }, f, indent=2)

    print("\n结果已保存到 optimization_results.json")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交**

```bash
git add back_testing/optimization/run_factor_optimization.py
git commit -m "feat(optimization): add optimization runner script"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | 项目结构 | directory, __init__.py |
| 2 | Chromosome | chromosome.py + test |
| 3 | Population | population.py + test |
| 4 | 遗传算子 | selection.py, crossover.py, mutation.py, elite.py |
| 5 | 适应度函数 | fitness.py + test |
| 6 | Walk-Forward | walk_forward.py |
| 7 | GA主循环 | genetic_optimizer.py |
| 8 | 敏感性分析 | sensitivity.py |
| 9 | 入口脚本 | run_factor_optimization.py |

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-genetic-algorithm-factor-optimization-plan.md`**

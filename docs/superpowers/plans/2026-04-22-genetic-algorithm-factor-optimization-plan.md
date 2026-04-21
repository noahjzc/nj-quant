# Genetic Algorithm Factor Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a genetic algorithm system to optimize 10 factor weights, maximizing Calmar ratio under 20% max drawdown constraint.

**Architecture:** Modular GA framework with chromosome encoding, population management, genetic operators, and walk-forward analysis. Uses simplified backtest (weekly rebalancing, equal-weight portfolio) for fast fitness evaluation during optimization.

**Tech Stack:** Python, pandas, numpy

---

## File Structure

```
back_testing/optimization/
├── __init__.py
├── genetic_optimizer/
│   ├── __init__.py
│   ├── chromosome.py       # 10-gene real-valued chromosome
│   ├── population.py      # Population management
│   ├── fitness.py         # Fitness evaluator with simplified backtest
│   ├── selection.py       # Tournament selection
│   ├── crossover.py       # Simulated Binary Crossover (SBX)
│   ├── mutation.py        # Gaussian mutation
│   ├── elite.py           # Elite preservation
│   ├── niching.py        # Fitness sharing (niching)
│   ├── walk_forward.py    # Walk-forward window generator
│   ├── sensitivity.py     # Factor sensitivity analysis
│   └── optimizer.py       # GA main loop
└── run_optimization.py    # Entry script

tests/back_testing/optimization/
├── __init__.py
├── test_chromosome.py
├── test_population.py
├── test_fitness.py
├── test_genetic_operators.py
└── test_optimizer.py
```

---

## Task 1: Project Structure and Chromosome Class

**Files:**
- Create: `back_testing/optimization/__init__.py`
- Create: `back_testing/optimization/genetic_optimizer/__init__.py`
- Create: `tests/back_testing/optimization/__init__.py`
- Create: `back_testing/optimization/genetic_optimizer/chromosome.py`
- Test: `tests/back_testing/optimization/test_chromosome.py`

- [ ] **Step 1: Create directory structure**

Run:
```bash
mkdir -p back_testing/optimization/genetic_optimizer
mkdir -p tests/back_testing/optimization
touch back_testing/optimization/__init__.py
touch back_testing/optimization/genetic_optimizer/__init__.py
touch tests/back_testing/optimization/__init__.py
```

- [ ] **Step 2: Write failing test for Chromosome**

Create file `tests/back_testing/optimization/test_chromosome.py`:

```python
"""Tests for Chromosome class"""
import pytest
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def test_chromosome_from_weights():
    """Test creating chromosome from weight dict"""
    weights = {'PB': 0.15, 'PE_TTM': 0.10, 'PS_TTM': 0.05, 'RSI_1': 0.15,
               'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
               'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05}
    chrom = Chromosome(weights)
    result = chrom.to_dict()

    # Verify weights are approximately correct
    assert abs(result['PB'] - 0.15) < 0.001
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_chromosome_auto_normalize():
    """Test that weights are auto-normalized to sum=1"""
    weights = {'PB': 0.5, 'PE_TTM': 0.5}  # Sum = 1.0
    # But with all 10 factors, sum will be different
    chrom = Chromosome({'PB': 0.3, 'PE_TTM': 0.3, 'PS_TTM': 0.1, 'RSI_1': 0.1,
                       'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05,
                       'TURNOVER': 0.05, 'VOLUME_RATIO': 0.0, 'AMPLITUDE': 0.0})
    result = chrom.to_dict()
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_chromosome_random_init():
    """Test random initialization"""
    chrom = Chromosome()
    result = chrom.to_dict()
    assert len(result) == 10
    assert abs(sum(result.values()) - 1.0) < 0.001
    assert all(0.01 <= v <= 0.40 for v in result.values())


def test_chromosome_copy():
    """Test chromosome deep copy"""
    chrom1 = Chromosome()
    chrom2 = chrom1.copy()
    assert chrom1.genes is not chrom2.genes
    assert np.allclose(chrom1.genes, chrom2.genes)


def test_chromosome_mutation():
    """Test that mutation preserves weight sum"""
    chrom = Chromosome()
    original_sum = sum(chrom.genes)
    chrom.mutate(gene_index=0, noise=0.1)
    assert abs(sum(chrom.genes) - 1.0) < 0.001


def test_chromosome_repr():
    """Test string representation"""
    chrom = Chromosome({'PB': 0.15, 'PE_TTM': 0.10, 'PS_TTM': 0.05, 'RSI_1': 0.15,
                       'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                       'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05})
    repr_str = repr(chrom)
    assert 'PB' in repr_str
    assert '0.15' in repr_str
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
pytest tests/back_testing/optimization/test_chromosome.py -v
```
Expected: FAIL - module not found

- [ ] **Step 4: Write Chromosome implementation**

Create file `back_testing/optimization/genetic_optimizer/chromosome.py`:

```python
"""Chromosome: Real-valued gene encoding for factor weights"""
import numpy as np
from typing import Dict


class Chromosome:
    """
    Chromosome encoding 10 factor weights as real-valued genes.

    Gene order: PB, PE_TTM, PS_TTM, RSI_1, KDJ_K, MA_5, MA_20,
                TURNOVER, VOLUME_RATIO, AMPLITUDE

    Constraints:
    - Each gene: [0.01, 0.40]
    - Sum of genes: 1.0 (normalized)
    """

    FACTOR_NAMES = [
        'PB', 'PE_TTM', 'PS_TTM', 'RSI_1', 'KDJ_K',
        'MA_5', 'MA_20', 'TURNOVER', 'VOLUME_RATIO', 'AMPLITUDE'
    ]

    MIN_WEIGHT = 0.01
    MAX_WEIGHT = 0.40

    def __init__(self, weights: Dict[str, float] = None, genes: np.ndarray = None):
        """
        Create chromosome.

        Args:
            weights: Weight dict {'factor': weight}
            genes: Direct gene array (for GA operations)
        """
        if weights is not None:
            self.genes = self._weights_to_genes(weights)
        elif genes is not None:
            self.genes = genes.copy()
        else:
            self.genes = self._random_init()

        self.fitness = None

    def _weights_to_genes(self, weights: Dict[str, float]) -> np.ndarray:
        """Convert weight dict to gene array and normalize"""
        genes = np.array([weights.get(f, 0.0) for f in self.FACTOR_NAMES])
        return self._normalize(genes)

    def _genes_to_weights(self) -> Dict[str, float]:
        """Convert gene array to weight dict"""
        return {f: w for f, w in zip(self.FACTOR_NAMES, self.genes)}

    def _normalize(self, genes: np.ndarray) -> np.ndarray:
        """Normalize genes to sum=1, clip to bounds"""
        # Clip to bounds first
        genes = np.clip(genes, self.MIN_WEIGHT, self.MAX_WEIGHT)
        # Normalize
        total = genes.sum()
        if total > 0:
            genes = genes / total
        else:
            genes = np.ones(len(genes)) / len(genes)
        return genes

    def _random_init(self) -> np.ndarray:
        """Random initialization with uniform distribution"""
        genes = np.random.rand(len(self.FACTOR_NAMES))
        return self._normalize(genes)

    def to_dict(self) -> Dict[str, float]:
        """Convert to weight dict"""
        return self._genes_to_weights()

    def copy(self) -> 'Chromosome':
        """Deep copy"""
        new_chrom = Chromosome(genes=self.genes)
        new_chrom.fitness = self.fitness
        return new_chrom

    def mutate(self, gene_index: int, noise: float = 0.1):
        """
        Mutate single gene with Gaussian noise.

        Args:
            gene_index: Index of gene to mutate
            noise: Standard deviation of Gaussian noise
        """
        self.genes[gene_index] += np.random.normal(0, noise)
        self.genes = self._normalize(self.genes)

    def __repr__(self):
        w = self.to_dict()
        return f"Chromosome({', '.join(f'{k}:{v:.3f}' for k, v in w.items())})"
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
pytest tests/back_testing/optimization/test_chromosome.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/chromosome.py
git add tests/back_testing/optimization/test_chromosome.py
git commit -m "feat(optimization): add Chromosome class for 10-factor weight encoding"
```

---

## Task 2: Population Class

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/population.py`
- Test: `tests/back_testing/optimization/test_population.py`

- [ ] **Step 1: Write failing test for Population**

Create file `tests/back_testing/optimization/test_population.py`:

```python
"""Tests for Population class"""
import pytest
from back_testing.optimization.genetic_optimizer.population import Population
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def test_population_creation():
    """Test population initialization"""
    pop = Population(size=50)
    assert len(pop.individuals) == 50
    assert pop.size == 50


def test_population_sort_by_fitness():
    """Test sorting population by fitness"""
    pop = Population(size=10)
    for i, chrom in enumerate(pop.individuals):
        chrom.fitness = i * 0.1

    pop.sort_by_fitness()

    for i in range(len(pop.individuals) - 1):
        assert pop.individuals[i].fitness >= pop.individuals[i + 1].fitness


def test_get_best():
    """Test getting best individual"""
    pop = Population(size=10)
    for i, chrom in enumerate(pop.individuals):
        chrom.fitness = i * 0.1

    best = pop.get_best()
    assert best.fitness == 0.9


def test_get_elite():
    """Test elite selection"""
    pop = Population(size=10)
    for i, chrom in enumerate(pop.individuals):
        chrom.fitness = i * 0.1

    elite = pop.get_elite(ratio=0.2)
    assert len(elite) == 2
    assert all(e.fitness >= 0.7 for e in elite)


def test_replace_individuals():
    """Test replacing population"""
    pop = Population(size=5)
    new_chroms = [Chromosome() for _ in range(3)]
    pop.replace_individuals(new_chroms)
    assert len(pop.individuals) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/back_testing/optimization/test_population.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Write Population implementation**

Create file `back_testing/optimization/genetic_optimizer/population.py`:

```python
"""Population management for genetic algorithm"""
from typing import List, Optional
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


class Population:
    """Population of chromosomes for genetic algorithm"""

    def __init__(self, size: int = 50):
        """
        Create population.

        Args:
            size: Number of individuals in population
        """
        self.size = size
        self.individuals: List[Chromosome] = []
        self._initialize()

    def _initialize(self):
        """Initialize with random chromosomes"""
        self.individuals = [Chromosome() for _ in range(self.size)]

    def sort_by_fitness(self, descending: bool = True):
        """Sort individuals by fitness"""
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
            reverse=descending
        )

    def get_best(self) -> Optional[Chromosome]:
        """Get best individual (highest fitness)"""
        if not self.individuals:
            return None
        self.sort_by_fitness()
        return self.individuals[0]

    def get_elite(self, ratio: float = 0.1) -> List[Chromosome]:
        """
        Get elite individuals.

        Args:
            ratio: Fraction of population to keep as elite

        Returns:
            List of elite chromosomes
        """
        n_elite = max(1, int(len(self.individuals) * ratio))
        self.sort_by_fitness()
        return [c.copy() for c in self.individuals[:n_elite]]

    def replace_individuals(self, new_individuals: List[Chromosome]):
        """Replace population with new individuals"""
        self.individuals = new_individuals[:self.size]

    def __len__(self):
        return len(self.individuals)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/back_testing/optimization/test_population.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/population.py
git add tests/back_testing/optimization/test_population.py
git commit -m "feat(optimization): add Population class for population management"
```

---

## Task 3: Genetic Operators (Selection, Crossover, Mutation, Elite)

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/selection.py`
- Create: `back_testing/optimization/genetic_optimizer/crossover.py`
- Create: `back_testing/optimization/genetic_optimizer/mutation.py`
- Create: `back_testing/optimization/genetic_optimizer/elite.py`
- Test: `tests/back_testing/optimization/test_genetic_operators.py`

- [ ] **Step 1: Write failing tests**

Create file `tests/back_testing/optimization/test_genetic_operators.py`:

```python
"""Tests for genetic operators"""
import pytest
import numpy as np
from back_testing.optimization.genetic_optimizer.selection import tournament_selection, select_parents
from back_testing.optimization.genetic_optimizer.crossover import crossover, simulated_binary_crossover
from back_testing.optimization.genetic_optimizer.mutation import gaussian_mutation, mutate_population
from back_testing.optimization.genetic_optimizer.elite import preserve_elite
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def test_tournament_selection():
    """Test tournament selection selects fittest"""
    pop = [Chromosome() for _ in range(10)]
    for i, chrom in enumerate(pop):
        chrom.fitness = i * 0.1

    # Run tournament many times, best should win most
    wins = {i: 0 for i in range(10)}
    for _ in range(100):
        winner = tournament_selection(pop, k=3)
        idx = pop.index(winner)
        wins[idx] += 1

    # Higher fitness individuals should win more
    assert wins[9] >= wins[5] >= wins[1]


def test_crossover_preserves_sum():
    """Test crossover produces valid offspring"""
    parent1 = Chromosome({'PB': 0.20, 'PE_TTM': 0.15, 'PS_TTM': 0.05, 'RSI_1': 0.15,
                         'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                         'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05})
    parent2 = Chromosome({'PB': 0.10, 'PE_TTM': 0.20, 'PS_TTM': 0.10, 'RSI_1': 0.10,
                         'KDJ_K': 0.10, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                         'VOLUME_RATIO': 0.10, 'AMPLITUDE': 0.05})

    child1, child2 = crossover(parent1, parent2, crossover_rate=1.0)

    # Both children should have valid weights
    assert abs(sum(child1.to_dict().values()) - 1.0) < 0.001
    assert abs(sum(child2.to_dict().values()) - 1.0) < 0.001


def test_mutation_preserves_sum():
    """Test mutation preserves weight sum constraint"""
    chrom = Chromosome()
    original_sum = sum(chrom.genes)

    mutated = gaussian_mutation(chrom, mutation_rate=1.0, noise=0.1)

    assert abs(sum(mutated.genes) - 1.0) < 0.001


def test_elite_preservation():
    """Test elite individuals are preserved"""
    pop = [Chromosome() for _ in range(10)]
    for i, chrom in enumerate(pop):
        chrom.fitness = i * 0.1

    elite = preserve_elite(pop, elite_ratio=0.2)

    assert len(elite) == 2
    assert elite[0].fitness == 0.9
    assert elite[1].fitness == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/back_testing/optimization/test_genetic_operators.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Write selection.py**

```python
"""Selection operator: Tournament selection"""
import numpy as np
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    """
    Tournament selection: select k individuals, return fittest.

    Args:
        population: List of chromosomes
        k: Tournament size

    Returns:
        Selected chromosome
    """
    tournament_idx = np.random.choice(len(population), size=k, replace=False)
    tournament = [(i, population[i].fitness if population[i].fitness is not None
                   else float('-inf')) for i in tournament_idx]
    winner_idx = max(tournament, key=lambda x: x[1])[0]
    return population[winner_idx].copy()


def select_parents(population: List[Chromosome], n_parents: int, k: int = 3) -> List[Chromosome]:
    """
    Select multiple parents via tournament selection.

    Args:
        population: List of chromosomes
        n_parents: Number of parents to select
        k: Tournament size

    Returns:
        List of selected parents
    """
    return [tournament_selection(population, k) for _ in range(n_parents)]
```

- [ ] **Step 4: Write crossover.py**

```python
"""Crossover operator: Simulated Binary Crossover (SBX)"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def simulated_binary_crossover(parent1: Chromosome, parent2: Chromosome,
                               eta: float = 20.0) -> tuple:
    """
    Simulated Binary Crossover (SBX).

    Args:
        parent1: First parent
        parent2: Second parent
        eta: Distribution index (higher = children closer to parents)

    Returns:
        (child1, child2)
    """
    n_genes = len(parent1.genes)
    child1_genes = np.zeros(n_genes)
    child2_genes = np.zeros(n_genes)

    for i in range(n_genes):
        if np.random.rand() < 0.5:
            if abs(parent1.genes[i] - parent2.genes[i]) > 1e-10:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

                child1_genes[i] = 0.5 * ((1 + beta) * parent1.genes[i] +
                                          (1 - beta) * parent2.genes[i])
                child2_genes[i] = 0.5 * ((1 - beta) * parent1.genes[i] +
                                          (1 + beta) * parent2.genes[i])
            else:
                child1_genes[i] = parent1.genes[i]
                child2_genes[i] = parent2.genes[i]
        else:
            child1_genes[i] = parent1.genes[i]
            child2_genes[i] = parent2.genes[i]

    child1 = Chromosome(genes=child1_genes)
    child2 = Chromosome(genes=child2_genes)

    return child1, child2


def crossover(parent1: Chromosome, parent2: Chromosome,
              crossover_rate: float = 0.7) -> tuple:
    """
    Crossover with probability.

    Args:
        parent1: First parent
        parent2: Second parent
        crossover_rate: Probability of crossover

    Returns:
        (child1, child2)
    """
    if np.random.rand() < crossover_rate:
        return simulated_binary_crossover(parent1, parent2)
    else:
        return parent1.copy(), parent2.copy()
```

- [ ] **Step 5: Write mutation.py**

```python
"""Mutation operator: Gaussian mutation"""
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def gaussian_mutation(chromosome: Chromosome,
                     mutation_rate: float = 0.05,
                     noise: float = 0.1) -> Chromosome:
    """
    Gaussian mutation with per-gene mutation rate.

    Args:
        chromosome: Individual to mutate
        mutation_rate: Probability of mutating each gene
        noise: Standard deviation of Gaussian noise

    Returns:
        Mutated chromosome
    """
    mutated = chromosome.copy()
    n_genes = len(mutated.genes)

    for i in range(n_genes):
        if np.random.rand() < mutation_rate:
            mutated.genes[i] += np.random.normal(0, noise)
            mutated.genes[i] = np.clip(mutated.genes[i],
                                        Chromosome.MIN_WEIGHT,
                                        Chromosome.MAX_WEIGHT)

    mutated.genes = mutated.genes / mutated.genes.sum()
    return mutated


def mutate_population(population: list,
                      mutation_rate: float = 0.05,
                      noise: float = 0.1) -> list:
    """
    Apply mutation to entire population.

    Args:
        population: List of chromosomes
        mutation_rate: Per-gene mutation rate
        noise: Gaussian noise standard deviation

    Returns:
        List of mutated chromosomes
    """
    return [gaussian_mutation(c, mutation_rate, noise) for c in population]
```

- [ ] **Step 6: Write elite.py**

```python
"""Elite preservation strategy"""
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def preserve_elite(population: List[Chromosome],
                   elite_ratio: float = 0.1) -> List[Chromosome]:
    """
    Select elite individuals (highest fitness).

    Args:
        population: Current population
        elite_ratio: Fraction to preserve as elite

    Returns:
        List of elite chromosomes
    """
    n_elite = max(1, int(len(population) * elite_ratio))

    sorted_pop = sorted(
        population,
        key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
        reverse=True
    )

    return [c.copy() for c in sorted_pop[:n_elite]]
```

- [ ] **Step 7: Run tests to verify they pass**

Run:
```bash
pytest tests/back_testing/optimization/test_genetic_operators.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/selection.py
git add back_testing/optimization/genetic_optimizer/crossover.py
git add back_testing/optimization/genetic_optimizer/mutation.py
git add back_testing/optimization/genetic_optimizer/elite.py
git add tests/back_testing/optimization/test_genetic_operators.py
git commit -m "feat(optimization): add genetic operators (selection, crossover, mutation, elite)"
```

---

## Task 4: Fitness Evaluator with Simplified Backtest

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/fitness.py`
- Test: `tests/back_testing/optimization/test_fitness.py`

- [ ] **Step 1: Write failing test**

Create file `tests/back_testing/optimization/test_fitness.py`:

```python
"""Tests for FitnessEvaluator"""
import pytest
import pandas as pd
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator


def test_fitness_evaluator_init():
    """Test FitnessEvaluator initialization"""
    evaluator = FitnessEvaluator(
        data_path='data/daily_ycz',
        max_drawdown_constraint=0.20,
        n_stocks=5
    )
    assert evaluator.max_drawdown_constraint == 0.20
    assert evaluator.n_stocks == 5


def test_get_rebalance_dates():
    """Test getting rebalance dates (Fridays)"""
    evaluator = FitnessEvaluator(data_path='data/daily_ycz')

    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2024-01-31')

    dates = evaluator._get_rebalance_dates(start, end)

    assert len(dates) >= 1
    for d in dates:
        assert d.weekday() == 4  # Friday


def test_empty_result():
    """Test empty result returns zeros"""
    evaluator = FitnessEvaluator(data_path='data/daily_ycz')
    result = evaluator._empty_result()

    assert result['annual_return'] == 0.0
    assert result['max_drawdown'] == 0.0
    assert result['total_return'] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/back_testing/optimization/test_fitness.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Write FitnessEvaluator implementation**

Create file `back_testing/optimization/genetic_optimizer/fitness.py`:

```python
"""
Fitness Evaluator: Simplified backtest for GA optimization.

Simplified vs full backtest:
- No daily ATR/stop-loss/take-profit (too slow for GA)
- Weekly rebalancing with equal-weight portfolio
- Uses real price data to compute actual returns
- Fast enough for thousands of GA evaluations
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.factors.factor_config import get_factor_directions
from back_testing.data.data_provider import DataProvider


class FitnessEvaluator:
    """
    Fitness evaluator using simplified weekly-rebalancing backtest.

    Optimization target: Calmar ratio = annual_return / max_drawdown
    Constraint: max_drawdown <= 20%
    """

    def __init__(self, data_path: str = None,
                 max_drawdown_constraint: float = 0.20,
                 n_stocks: int = 5):
        """
        Args:
            data_path: Path to data directory (Parquet/CSV)
            max_drawdown_constraint: Maximum allowed drawdown
            n_stocks: Number of stocks to hold
        """
        self.data_path = data_path
        self.max_drawdown_constraint = max_drawdown_constraint
        self.n_stocks = n_stocks

        self.data_provider = DataProvider(use_db=False, data_dir=data_path)
        self.factor_loader = FactorLoader(data_provider=self.data_provider)

    def evaluate(self, weights: Dict[str, float],
                start_date: pd.Timestamp,
                end_date: pd.Timestamp) -> float:
        """
        Evaluate fitness of a weight configuration.

        Args:
            weights: Factor weights dict
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Calmar ratio (0 if constraint violated)
        """
        try:
            result = self._run_backtest(weights, start_date, end_date)

            annual_return = result.get('annual_return', 0)
            max_drawdown = result.get('max_drawdown', 0)

            if max_drawdown > self.max_drawdown_constraint:
                return 0.0

            # Prevent division by zero
            calmar = annual_return / max(max_drawdown, 0.01)

            return calmar

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0

    def _run_backtest(self, weights: Dict[str, float],
                     start_date: pd.Timestamp,
                     end_date: pd.Timestamp) -> Dict:
        """
        Run simplified backtest.

        Weekly flow:
        1. Get rebalance dates (Fridays)
        2. Each Friday: select stocks by factor scoring
        3. Next Friday: calculate holding period return
        4. Accumulate portfolio value curve
        5. Calculate performance metrics
        """
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        if len(rebalance_dates) < 2:
            return self._empty_result()

        factor_directions = get_factor_directions()

        portfolio_values = [1.0]

        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            factor_list = list(weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(
                current_date, factor_list
            )

            if len(factor_data) == 0:
                portfolio_values.append(portfolio_values[-1])
                continue

            selector = MultiFactorSelector(
                weights=weights,
                directions=factor_directions
            )
            selected_stocks = selector.select_top_stocks(
                data=factor_data,
                n=self.n_stocks
            )

            if not selected_stocks:
                portfolio_values.append(portfolio_values[-1])
                continue

            period_return = self._calculate_period_return(
                selected_stocks, current_date, next_date
            )

            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

        portfolio_values = np.array(portfolio_values)
        total_return = portfolio_values[-1] / portfolio_values[0] - 1

        n_weeks = len(portfolio_values) - 1
        if n_weeks > 0:
            annual_return = (1 + total_return) ** (52 / n_weeks) - 1
        else:
            annual_return = 0

        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'n_weeks': n_weeks
        }

    def _get_rebalance_dates(self, start_date: pd.Timestamp,
                            end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Get all Fridays between start and end dates"""
        dates = []
        current = pd.Timestamp(start_date)

        while current.weekday() != 4:
            current += timedelta(days=1)

        while current <= end_date:
            dates.append(current)
            current += timedelta(days=7)

        return dates

    def _calculate_period_return(self, stocks: List[str],
                                 current_date: pd.Timestamp,
                                 next_date: pd.Timestamp) -> float:
        """
        Calculate equal-weighted return of selected stocks over period.

        Args:
            stocks: List of stock codes
            current_date: Rebalance date
            next_date: Next rebalance date

        Returns:
            Equal-weighted portfolio return
        """
        returns = []

        for stock in stocks:
            try:
                df = self.data_provider.get_stock_data(
                    stock,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=next_date.strftime('%Y-%m-%d')
                )

                if len(df) < 2:
                    continue

                price_col = None
                for col in ['后复权价', 'close', '收盘价']:
                    if col in df.columns:
                        price_col = col
                        break

                if price_col is None:
                    continue

                df = df.sort_index()
                prices = df[price_col].values

                if len(prices) >= 2:
                    period_return = (prices[-1] / prices[0]) - 1
                    returns.append(period_return)

            except Exception:
                continue

        if not returns:
            return 0.0

        return np.mean(returns)

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown from value sequence"""
        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _empty_result(self) -> Dict:
        """Return empty result for invalid evaluation"""
        return {
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'n_weeks': 0
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/back_testing/optimization/test_fitness.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/fitness.py
git add tests/back_testing/optimization/test_fitness.py
git commit -m "feat(optimization): add FitnessEvaluator with simplified backtest"
```

---

## Task 5: Walk-Forward Analyzer

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/walk_forward.py`

- [ ] **Step 1: Write implementation**

Create file `back_testing/optimization/genetic_optimizer/walk_forward.py`:

```python
"""Walk-Forward Analysis for robust optimization"""
import pandas as pd
from typing import List, Dict, Tuple


class WalkForwardAnalyzer:
    """
    Walk-Forward window generator and weight aggregator.

    Window structure:
    - Train: 3 years (GA optimization)
    - Validation: 1 year (early stopping)
    - Test: 1 year (final evaluation)

    Rolls forward every 3 months.
    """

    def __init__(self,
                 train_window_years: int = 3,
                 val_window_years: int = 1,
                 test_window_years: int = 1,
                 step_months: int = 3):
        """
        Args:
            train_window_years: Training window length in years
            val_window_years: Validation window length in years
            test_window_years: Test window length in years
            step_months: Rolling step in months
        """
        self.train_window_years = train_window_years
        self.val_window_years = val_window_years
        self.test_window_years = test_window_years
        self.step_months = step_months

    def get_windows(self, start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> List[Dict]:
        """
        Generate walk-forward windows.

        Args:
            start_date: Data start date
            end_date: Data end date

        Returns:
            List of windows: [{'train': (start, end), 'val': ..., 'test': ...}, ...]
        """
        windows = []
        current = pd.Timestamp(start_date)

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
        Aggregate weights from multiple windows.

        Args:
            weights_list: List of optimal weights from each window

        Returns:
            Aggregated weights (simple average)
        """
        if not weights_list:
            return {}

        n = len(weights_list)
        aggregated = {}
        all_keys = set()
        for w in weights_list:
            all_keys.update(w.keys())

        for key in all_keys:
            values = [w.get(key, 0) for w in weights_list]
            aggregated[key] = sum(values) / n

        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return aggregated
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/walk_forward.py
git commit -m "feat(optimization): add WalkForwardAnalyzer for walk-forward analysis"
```

---

## Task 6: GA Main Loop (Optimizer)

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/optimizer.py`
- Test: `tests/back_testing/optimization/test_optimizer.py`

- [ ] **Step 1: Write failing test**

Create file `tests/back_testing/optimization/test_optimizer.py`:

```python
"""Tests for GeneticOptimizer"""
import pytest
from back_testing.optimization.genetic_optimizer.optimizer import GeneticOptimizer


def test_optimizer_init():
    """Test optimizer initialization with default params"""
    optimizer = GeneticOptimizer()
    assert optimizer.population_size == 50
    assert optimizer.max_generations == 100
    assert optimizer.elite_ratio == 0.1
    assert optimizer.crossover_rate == 0.7
    assert optimizer.mutation_rate == 0.05
    assert optimizer.tournament_k == 3
    assert optimizer.patience == 20


def test_optimizer_custom_params():
    """Test optimizer with custom parameters"""
    optimizer = GeneticOptimizer(
        population_size=100,
        max_generations=50,
        elite_ratio=0.2
    )
    assert optimizer.population_size == 100
    assert optimizer.max_generations == 50
    assert optimizer.elite_ratio == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/back_testing/optimization/test_optimizer.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Write optimizer implementation**

Create file `back_testing/optimization/genetic_optimizer/optimizer.py`:

```python
"""Genetic Algorithm Optimizer: Main loop"""
import numpy as np
from typing import Dict, Callable, Optional
from back_testing.optimization.genetic_optimizer.population import Population
from back_testing.optimization.genetic_optimizer.selection import tournament_selection
from back_testing.optimization.genetic_optimizer.crossover import crossover
from back_testing.optimization.genetic_optimizer.mutation import gaussian_mutation
from back_testing.optimization.genetic_optimizer.elite import preserve_elite


class GeneticOptimizer:
    """
    Genetic Algorithm for factor weight optimization.

    Process:
    1. Initialize population
    2. Evaluate fitness
    3. Select, crossover, mutate
    4. Preserve elite
    5. Repeat until convergence or max generations
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
        Args:
            population_size: Number of individuals
            max_generations: Maximum iterations
            elite_ratio: Fraction of best individuals to preserve
            crossover_rate: Probability of crossover
            mutation_rate: Per-gene mutation probability
            tournament_k: Tournament size for selection
            patience: Generations without improvement before early stop
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.patience = patience

        if seed is not None:
            np.random.seed(seed)

    def optimize(self,
                fitness_func: Callable,
                train_data,
                val_data: Optional = None,
                verbose: bool = True) -> Dict[str, float]:
        """
        Run genetic algorithm optimization.

        Args:
            fitness_func: Fitness function (weights -> float)
            train_data: Training data passed to fitness_func
            val_data: Validation data for early stopping (optional)
            verbose: Print progress

        Returns:
            Optimal weights dict
        """
        population = Population(size=self.population_size)

        for chrom in population.individuals:
            chrom.fitness = fitness_func(chrom.to_dict(), train_data)

        elite = preserve_elite(population.individuals, self.elite_ratio)

        no_improve_count = 0
        best_val_fitness = float('-inf')
        best_weights = None

        for gen in range(self.max_generations):
            new_individuals = []
            new_individuals.extend(elite)

            while len(new_individuals) < self.population_size:
                parent1 = tournament_selection(population.individuals, self.tournament_k)
                parent2 = tournament_selection(population.individuals, self.tournament_k)

                child1, child2 = crossover(parent1, parent2, self.crossover_rate)

                child1 = gaussian_mutation(child1, self.mutation_rate)
                child2 = gaussian_mutation(child2, self.mutation_rate)

                child1.fitness = fitness_func(child1.to_dict(), train_data)
                child2.fitness = fitness_func(child2.to_dict(), train_data)

                new_individuals.extend([child1, child2])

            population.replace_individuals(new_individuals[:self.population_size])
            elite = preserve_elite(population.individuals, self.elite_ratio)

            if val_data is not None:
                current_best = max(
                    c.fitness for c in population.individuals
                    if c.fitness is not None
                )
                if current_best > best_val_fitness:
                    best_val_fitness = current_best
                    no_improve_count = 0
                    best_weights = population.get_best().to_dict()
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    if verbose:
                        print(f"Early stop at generation {gen + 1}")
                    break

            if verbose and (gen + 1) % 10 == 0:
                best = population.get_best()
                print(f"Gen {gen + 1}: fitness = {best.fitness if best else 'N/A'}")

        if best_weights is None:
            best_weights = population.get_best().to_dict()

        return best_weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/back_testing/optimization/test_optimizer.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/optimizer.py
git add tests/back_testing/optimization/test_optimizer.py
git commit -m "feat(optimization): add GeneticOptimizer main loop"
```

---

## Task 7: Sensitivity Analyzer

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/sensitivity.py`

- [ ] **Step 1: Write implementation**

Create file `back_testing/optimization/genetic_optimizer/sensitivity.py`:

```python
"""Sensitivity Analysis: Factor importance evaluation"""
import numpy as np
import pandas as pd
from typing import Dict


class SensitivityAnalyzer:
    """
    Analyze factor importance by testing weight sensitivity.

    Method:
    1. For each factor, vary its weight while keeping others fixed
    2. Measure change in Calmar ratio
    3. Higher sensitivity = more important factor
    """

    def __init__(self, evaluator):
        """
        Args:
            evaluator: FitnessEvaluator instance
        """
        self.evaluator = evaluator

    def analyze_factor_importance(self,
                                  optimal_weights: Dict[str, float],
                                  data,
                                  factor_range: tuple = (0.0, 0.5),
                                  steps: int = 10) -> pd.DataFrame:
        """
        Analyze importance of each factor.

        Args:
            optimal_weights: Optimal weights from GA
            data: (start_date, end_date) tuple for evaluation
            factor_range: Weight range to test
            steps: Number of steps in range

        Returns:
            DataFrame with factor importance scores
        """
        results = []

        for factor, base_weight in optimal_weights.items():
            sensitivities = []

            for pct in np.linspace(factor_range[0], factor_range[1], steps):
                test_weights = optimal_weights.copy()
                test_weights[factor] = pct

                total = sum(test_weights.values())
                test_weights = {k: v / total for k, v in test_weights.items()}

                fitness = self.evaluator.evaluate(test_weights, data[0], data[1])
                sensitivities.append(fitness)

            sensitivity = np.std(sensitivities)
            results.append({
                'factor': factor,
                'base_weight': base_weight,
                'sensitivity': sensitivity,
                'fitness_range': max(sensitivities) - min(sensitivities)
            })

        return pd.DataFrame(results).sort_values('sensitivity', ascending=False)
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/sensitivity.py
git commit -m "feat(optimization): add SensitivityAnalyzer for factor importance"
```

---

## Task 8: Niching (Fitness Sharing)

**Files:**
- Create: `back_testing/optimization/genetic_optimizer/niching.py`

- [ ] **Step 1: Write implementation**

Create file `back_testing/optimization/genetic_optimizer/niching.py`:

```python
"""Niching: Fitness sharing to maintain population diversity"""
import numpy as np
from typing import List
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def calculate_distance(chrom1: Chromosome, chrom2: Chromosome) -> float:
    """
    Calculate Euclidean distance between two chromosomes.

    Args:
        chrom1: First chromosome
        chrom2: Second chromosome

    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((chrom1.genes - chrom2.genes) ** 2))


def apply_niching(population: List[Chromosome],
                  sigma: float = 0.1,
                  alpha: float = 1.0) -> List[Chromosome]:
    """
    Apply fitness sharing to reduce fitness of similar individuals.

    Args:
        population: Current population
        sigma: Sharing radius (distance threshold)
        alpha: Sharing strength

    Returns:
        Population with adjusted fitness
    """
    for i, chrom_i in enumerate(population):
        if chrom_i.fitness is None:
            continue

        niche_count = 0.0

        for j, chrom_j in enumerate(population):
            if i == j or chrom_j.fitness is None:
                continue

            distance = calculate_distance(chrom_i, chrom_j)

            if distance < sigma:
                sharing = 1.0 - (distance / sigma) ** alpha
                niche_count += sharing

        if niche_count > 0:
            population[i].fitness = chrom_i.fitness / niche_count

    return population
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/genetic_optimizer/niching.py
git commit -m "feat(optimization): add niching for population diversity"
```

---

## Task 9: Entry Script

**Files:**
- Create: `back_testing/optimization/run_optimization.py`

- [ ] **Step 1: Write entry script**

Create file `back_testing/optimization/run_optimization.py`:

```python
"""Factor Weight Optimization Entry Script"""
import pandas as pd
import json
from back_testing.optimization.genetic_optimizer.optimizer import GeneticOptimizer
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator
from back_testing.optimization.genetic_optimizer.walk_forward import WalkForwardAnalyzer
from back_testing.optimization.genetic_optimizer.sensitivity import SensitivityAnalyzer


def main():
    """Run factor weight optimization"""
    # Configuration
    DATA_PATH = 'data/daily_ycz'
    START_DATE = pd.Timestamp('2019-01-01')
    END_DATE = pd.Timestamp('2024-01-01')

    print("=" * 60)
    print("Genetic Algorithm Factor Weight Optimization")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Period: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    # Create evaluator
    evaluator = FitnessEvaluator(
        data_path=DATA_PATH,
        max_drawdown_constraint=0.20,
        n_stocks=5
    )

    # Create walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(
        train_window_years=3,
        val_window_years=1,
        test_window_years=1,
        step_months=3
    )

    windows = wf_analyzer.get_windows(START_DATE, END_DATE)
    print(f"\nWalk-Forward Windows: {len(windows)}")

    all_optimal_weights = []

    for i, window in enumerate(windows):
        print(f"\n=== Window {i + 1}/{len(windows)} ===")
        print(f"Train: {window['train'][0].date()} ~ {window['train'][1].date()}")
        print(f"Val:   {window['val'][0].date()} ~ {window['val'][1].date()}")
        print(f"Test:  {window['test'][0].date()} ~ {window['test'][1].date()}")

        def fitness_func(weights, data):
            return evaluator.evaluate(weights, data[0], data[1])

        optimizer = GeneticOptimizer(
            population_size=50,
            max_generations=100,
            elite_ratio=0.1,
            crossover_rate=0.7,
            mutation_rate=0.05,
            patience=20,
            seed=42
        )

        optimal = optimizer.optimize(
            fitness_func=lambda w: fitness_func(w, (window['train'][0], window['train'][1])),
            train_data=(window['train'][0], window['train'][1]),
            val_data=(window['val'][0], window['val'][1]),
            verbose=True
        )

        val_fitness = evaluator.evaluate(optimal, window['val'][0], window['val'][1])
        test_fitness = evaluator.evaluate(optimal, window['test'][0], window['test'][1])

        print(f"Val Calmar: {val_fitness:.4f}")
        print(f"Test Calmar: {test_fitness:.4f}")

        all_optimal_weights.append(optimal)

    final_weights = wf_analyzer.aggregate_weights(all_optimal_weights)

    print("\n" + "=" * 60)
    print("Final Aggregated Weights")
    print("=" * 60)
    for k, v in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    print("\n=== Sensitivity Analysis ===")
    sensitivity = SensitivityAnalyzer(evaluator)
    importance = sensitivity.analyze_factor_importance(
        final_weights,
        (START_DATE, END_DATE)
    )
    print(importance.to_string(index=False))

    results = {
        'final_weights': final_weights,
        'window_weights': all_optimal_weights,
        'factor_importance': importance.to_dict()
    }

    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to optimization_results.json")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/run_optimization.py
git commit -m "feat(optimization): add optimization runner script"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project structure + Chromosome | chromosome.py + test |
| 2 | Population class | population.py + test |
| 3 | Genetic operators | selection.py, crossover.py, mutation.py, elite.py + test |
| 4 | Fitness evaluator | fitness.py + test |
| 5 | Walk-Forward analyzer | walk_forward.py |
| 6 | GA optimizer | optimizer.py + test |
| 7 | Sensitivity analyzer | sensitivity.py |
| 8 | Niching | niching.py |
| 9 | Entry script | run_optimization.py |

**Plan complete and saved to `docs/superpowers/plans/2026-04-22-genetic-algorithm-factor-optimization-plan.md`**

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

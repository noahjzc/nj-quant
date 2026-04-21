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

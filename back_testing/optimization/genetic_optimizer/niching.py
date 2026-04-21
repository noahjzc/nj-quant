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
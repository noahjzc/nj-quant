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

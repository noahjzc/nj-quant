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

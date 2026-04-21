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
        """Replace population with new individuals, padding from existing if needed"""
        if len(new_individuals) >= self.size:
            self.individuals = new_individuals[:self.size]
        else:
            # Save current individuals before replacing
            current = list(self.individuals)
            self.individuals = list(new_individuals)
            remaining = self.size - len(new_individuals)
            # Sort current by fitness (descending) to get best
            current.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
            # Fill remaining slots with best from current
            self.individuals = list(new_individuals) + [c.copy() for c in current[:remaining]]

    def __len__(self):
        return len(self.individuals)

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

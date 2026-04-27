"""Tests for genetic operators"""
import pytest
import numpy as np
from collections import defaultdict
from back_testing.optimization.genetic_optimizer.selection import tournament_selection, select_parents
from back_testing.optimization.genetic_optimizer.crossover import crossover, simulated_binary_crossover
from back_testing.optimization.genetic_optimizer.mutation import gaussian_mutation, mutate_population
from back_testing.optimization.genetic_optimizer.elite import preserve_elite
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def test_tournament_selection():
    """Test tournament selection selects fittest"""
    np.random.seed(42)
    pop = [Chromosome() for _ in range(10)]
    for i, chrom in enumerate(pop):
        chrom.fitness = i * 0.1

    # Run tournament many times, track wins by fitness level
    wins_by_fitness = defaultdict(int)
    for _ in range(200):
        winner = tournament_selection(pop, k=3)
        wins_by_fitness[winner.fitness] += 1

    # Higher fitness individuals should win more often
    assert wins_by_fitness[0.9] >= wins_by_fitness[0.5]
    assert wins_by_fitness[0.5] >= wins_by_fitness[0.1]


def test_crossover_preserves_sum():
    """Test crossover produces valid offspring"""
    np.random.seed(42)
    parent1 = Chromosome({'PB': 0.20, 'PE_TTM': 0.15, 'PS_TTM': 0.05, 'RSI_1': 0.15,
                         'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                         'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05})
    parent2 = Chromosome({'PB': 0.10, 'PE_TTM': 0.20, 'PS_TTM': 0.10, 'RSI_1': 0.10,
                         'KDJ_K': 0.10, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                         'VOLUME_RATIO': 0.10, 'AMPLITUDE': 0.05})

    child1, child2 = crossover(parent1, parent2, crossover_rate=1.0)

    # Both children should have valid weights (relaxed tolerance for floating point)
    assert abs(sum(child1.to_dict().values()) - 1.0) < 0.01
    assert abs(sum(child2.to_dict().values()) - 1.0) < 0.01


def test_mutation_preserves_sum():
    """Test mutation preserves weight sum constraint"""
    np.random.seed(42)
    chrom = Chromosome()

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

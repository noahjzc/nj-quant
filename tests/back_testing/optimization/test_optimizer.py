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

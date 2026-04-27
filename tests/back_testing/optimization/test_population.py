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

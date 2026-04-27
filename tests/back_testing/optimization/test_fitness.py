"""Tests for FitnessEvaluator"""
import pytest
import pandas as pd
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator


def test_fitness_evaluator_init():
    """Test FitnessEvaluator initialization"""
    evaluator = FitnessEvaluator(
        data_path='data/daily_ycz',
        max_drawdown_constraint=0.20,
        n_stocks=5
    )
    assert evaluator.max_drawdown_constraint == 0.20
    assert evaluator.n_stocks == 5


def test_get_rebalance_dates():
    """Test getting rebalance dates (Fridays)"""
    evaluator = FitnessEvaluator(data_path='data/daily_ycz')

    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2024-01-31')

    dates = evaluator._get_rebalance_dates(start, end)

    assert len(dates) >= 1
    for d in dates:
        assert d.weekday() == 4  # Friday


def test_empty_result():
    """Test empty result returns zeros"""
    evaluator = FitnessEvaluator(data_path='data/daily_ycz')
    result = evaluator._empty_result()

    assert result['annual_return'] == 0.0
    assert result['max_drawdown'] == 0.0
    assert result['total_return'] == 0.0
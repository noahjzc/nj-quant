"""Tests for new Optuna optimization parameters."""
import optuna
from optimization.optuna.run_daily_rotation_optimization import sample_config


def test_sample_config_includes_new_factors():
    study = optuna.create_study()
    trial = study.ask()
    config = sample_config(trial)
    assert 'circulating_mv' in config.rank_factor_weights
    assert 'WR_10' in config.rank_factor_weights
    assert 'WR_14' in config.rank_factor_weights
    assert 'circulating_mv' in config.rank_factor_directions
    assert config.rank_factor_directions['circulating_mv'] == -1


def test_sample_config_has_kdj_low_threshold():
    study = optuna.create_study()
    trial = study.ask()
    config = sample_config(trial)
    assert hasattr(config, 'kdj_low_threshold')
    assert 20.0 <= config.kdj_low_threshold <= 40.0


def test_kdj_gold_low_in_signal_pool():
    from optimization.optuna.run_daily_rotation_optimization import ALL_SIGNAL_TYPES
    assert 'KDJ_GOLD_LOW' in ALL_SIGNAL_TYPES

"""Tests for Chromosome class"""
import pytest
import numpy as np
from back_testing.optimization.genetic_optimizer.chromosome import Chromosome


def test_chromosome_from_weights():
    """Test creating chromosome from weight dict"""
    weights = {'PB': 0.15, 'PE_TTM': 0.10, 'PS_TTM': 0.05, 'RSI_1': 0.15,
               'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
               'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05}
    chrom = Chromosome(weights)
    result = chrom.to_dict()

    # Verify weights are approximately correct
    assert abs(result['PB'] - 0.15) < 0.001
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_chromosome_auto_normalize():
    """Test that weights are auto-normalized to sum=1"""
    chrom = Chromosome({'PB': 0.3, 'PE_TTM': 0.3, 'PS_TTM': 0.1, 'RSI_1': 0.1,
                       'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05,
                       'TURNOVER': 0.05, 'VOLUME_RATIO': 0.0, 'AMPLITUDE': 0.0})
    result = chrom.to_dict()
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_chromosome_random_init():
    """Test random initialization"""
    chrom = Chromosome()
    result = chrom.to_dict()
    assert len(result) == 10
    assert abs(sum(result.values()) - 1.0) < 0.001
    assert all(0.01 <= v <= 0.40 for v in result.values())


def test_chromosome_copy():
    """Test chromosome deep copy"""
    chrom1 = Chromosome()
    chrom2 = chrom1.copy()
    assert chrom1.genes is not chrom2.genes
    assert np.allclose(chrom1.genes, chrom2.genes)


def test_chromosome_mutation():
    """Test that mutation preserves weight sum"""
    chrom = Chromosome()
    chrom.mutate(gene_index=0, noise=0.1)
    assert abs(sum(chrom.genes) - 1.0) < 0.001


def test_chromosome_repr():
    """Test string representation"""
    chrom = Chromosome({'PB': 0.15, 'PE_TTM': 0.10, 'PS_TTM': 0.05, 'RSI_1': 0.15,
                       'KDJ_K': 0.05, 'MA_5': 0.05, 'MA_20': 0.05, 'TURNOVER': 0.10,
                       'VOLUME_RATIO': 0.05, 'AMPLITUDE': 0.05})
    repr_str = repr(chrom)
    assert 'PB' in repr_str
    assert '0.15' in repr_str
"""Chromosome: Real-valued gene encoding for factor weights"""
import numpy as np
from typing import Dict


class Chromosome:
    """
    Chromosome encoding 10 factor weights as real-valued genes.

    Gene order: PB, PE_TTM, PS_TTM, RSI_1, KDJ_K, MA_5, MA_20,
                TURNOVER, VOLUME_RATIO, AMPLITUDE

    Constraints:
    - Each gene: [0.01, 0.40]
    - Sum of genes: 1.0 (normalized)
    """

    FACTOR_NAMES = [
        'PB', 'PE_TTM', 'PS_TTM', 'RSI_1', 'KDJ_K',
        'MA_5', 'MA_20', 'TURNOVER', 'VOLUME_RATIO', 'AMPLITUDE'
    ]

    MIN_WEIGHT = 0.01
    MAX_WEIGHT = 0.40

    def __init__(self, weights: Dict[str, float] = None, genes: np.ndarray = None):
        """
        Create chromosome.

        Args:
            weights: Weight dict {'factor': weight}
            genes: Direct gene array (for GA operations)
        """
        if weights is not None:
            self.genes = self._weights_to_genes(weights)
        elif genes is not None:
            self.genes = genes.copy()
        else:
            self.genes = self._random_init()

        self.fitness = None

    def _weights_to_genes(self, weights: Dict[str, float]) -> np.ndarray:
        """Convert weight dict to gene array and normalize"""
        genes = np.array([weights.get(f, 0.0) for f in self.FACTOR_NAMES])
        return self._normalize(genes)

    def _genes_to_weights(self) -> Dict[str, float]:
        """Convert gene array to weight dict"""
        return {f: w for f, w in zip(self.FACTOR_NAMES, self.genes)}

    def _normalize(self, genes: np.ndarray) -> np.ndarray:
        """Normalize genes to sum=1, clip to bounds"""
        # Clip to bounds first
        genes = np.clip(genes, self.MIN_WEIGHT, self.MAX_WEIGHT)
        # Normalize
        total = genes.sum()
        if total > 0:
            genes = genes / total
        else:
            genes = np.ones(len(genes)) / len(genes)
        return genes

    def _random_init(self) -> np.ndarray:
        """Random initialization with uniform distribution"""
        genes = np.random.rand(len(self.FACTOR_NAMES))
        return self._normalize(genes)

    def to_dict(self) -> Dict[str, float]:
        """Convert to weight dict"""
        return self._genes_to_weights()

    def copy(self) -> 'Chromosome':
        """Deep copy"""
        new_chrom = Chromosome(genes=self.genes)
        new_chrom.fitness = self.fitness
        return new_chrom

    def mutate(self, gene_index: int, noise: float = 0.1):
        """
        Mutate single gene with Gaussian noise.

        Args:
            gene_index: Index of gene to mutate
            noise: Standard deviation of Gaussian noise
        """
        self.genes[gene_index] += np.random.normal(0, noise)
        self.genes = self._normalize(self.genes)

    def __repr__(self):
        w = self.to_dict()
        return f"Chromosome({', '.join(f'{k}:{v:.3f}' for k, v in w.items())})"

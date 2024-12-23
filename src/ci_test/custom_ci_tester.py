# === IMPORTS: BUILT-IN ===
from abc import ABC, abstractmethod
from typing import List

# === IMPORTS: THIRD-PARTY ===
import numpy as np


class ConditionalIndependenceTester(ABC):
    def __init__(self, data: np.ndarray, epsilon: float, alphabet_sizes: List[int]):
        self.data = data
        self.epsilon = epsilon
        self.alphabet_sizes = alphabet_sizes

    """
    Returns whether sum_{a,b,c} P(c) * | P(a,b|c) - P(a|c) * P(b|c) | <= eps
    """
    @abstractmethod
    def test(self, A: set, B: set, C: set) -> dict:
        raise NotImplementedError
        
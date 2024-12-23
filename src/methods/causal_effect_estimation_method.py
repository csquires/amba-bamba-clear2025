# === IMPORTS: BUILT-IN ===
from abc import ABC, abstractmethod
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.utils import Query
from src.problem_instance import ProblemInstance


@dataclass
class MethodHyperparameters:
    pass
        


class CausalEffectEstimationMethod(ABC):
    def __init__(
        self,
        hyperparameters: MethodHyperparameters
    ):
        self.hyperparameters = hyperparameters
        
    @abstractmethod
    def name(self):
        raise NotImplementedError
    
    @abstractmethod
    def key(self):
        raise NotImplementedError
    
    @abstractmethod
    def color(self):
        return "k"

    @abstractmethod
    def estimate(self, problem_instance: ProblemInstance, query: Query):
        raise NotImplementedError
        
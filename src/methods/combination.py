# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.synthetic_generation import ProblemInstance
from src.utils import Query
from src.methods.bamba import Bamba
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters


@dataclass
class CombinationHyperparameters(MethodHyperparameters):
    epsilon2: float
    conditional_independence_tester_class: type
    alpha: float = 0

class Combination(CausalEffectEstimationMethod):
    def __init__(self, hyperparameters: CombinationHyperparameters):
        super().__init__(hyperparameters)
        self.hyperparameters = hyperparameters
    
    @property
    def name(self):
        return "Combination"
    
    @property
    def key(self):
        return f"Combination(alpha={self.hyperparameters.alpha}, epsilon2={self.hyperparameters.epsilon2})"
    
    @property
    def color(self):
        return "g"

    def estimate(
            self, 
            problem_instance: ProblemInstance, 
            query: Query
        ) -> float:
            bamba = Bamba(self.hyperparameters)
            return bamba.estimate(problem_instance, query, combo_check=True)
            
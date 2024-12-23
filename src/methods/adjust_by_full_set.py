# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters
from src.synthetic_generation import ProblemInstance
from src.utils import Query, adjust_by_S


@dataclass
class AdjustByFullSetHyperparameters(MethodHyperparameters):
    alpha: float = 0


class AdjustByFullSet(CausalEffectEstimationMethod):
    @property
    def name(self):
        return "Z-Adjust"
    
    @property
    def key(self):
        return f"Z-Adjust(alpha={self.hyperparameters.alpha})"
    
    @property
    def color(self):
        return "b"
    
    def estimate(self, problem_instance: ProblemInstance, query: Query) -> float:
        return adjust_by_S(
            problem_instance.samples, 
            query, 
            problem_instance.nondescendants, 
            problem_instance.alphabet_sizes,
            alpha=self.hyperparameters.alpha
        )

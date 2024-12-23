# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters
from src.synthetic_generation import ProblemInstance
from src.utils import Query, adjust_by_S


@dataclass
class AdjustByMarkovBlanketHyperparameters(MethodHyperparameters):
    alpha: float = 0


class AdjustByMarkovBlanket(CausalEffectEstimationMethod):
    @property
    def name(self):
        return "MB-Adjust"
    
    @property
    def key(self):
        return f"MB-Adjust(alpha={self.hyperparameters.alpha})"
    
    @property
    def color(self):
        return "r"

    def estimate(self, problem_instance: ProblemInstance, query: Query) -> float:
        return adjust_by_S(
            problem_instance.samples, 
            query, 
            problem_instance.parents | problem_instance.spouses, 
            problem_instance.alphabet_sizes,
            alpha=self.hyperparameters.alpha
        )
    
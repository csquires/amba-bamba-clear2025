# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters
from src.synthetic_generation import ProblemInstance
from src.utils import Query, adjust_by_S, get_minimal_adjset


@dataclass
class AdjustByMinimalSetHyperparameters(MethodHyperparameters):
    alpha: float = 0


class AdjustByMinimalSet(CausalEffectEstimationMethod):
    @property
    def name(self):
        return "Min-Adjust"
    
    @property
    def key(self):
        return f"Min-Adjust(alpha={self.hyperparameters.alpha})"
    
    @property
    def color(self):
        return "b"
    
    def estimate(self, problem_instance: ProblemInstance, query: Query) -> float:
        minimal_adjustment_set = get_minimal_adjset(problem_instance.dag, query.action, query.target)
        return adjust_by_S(
            problem_instance.samples, 
            query, 
            minimal_adjustment_set, 
            problem_instance.alphabet_sizes,
            alpha=self.hyperparameters.alpha
        )

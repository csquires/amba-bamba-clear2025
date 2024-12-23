# === IMPORTS: BUILT-IN ===
import itertools as itr
from time import time
from dataclasses import dataclass

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.synthetic_generation import ProblemInstance
from src.ci_test.custom_ci_tester import ConditionalIndependenceTester
from src.utils import Query, adjust_by_S, estimate_alpha_lower_bound
from src.methods.amba import find_approx_markov_blanket
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters


@dataclass
class BambaHyperparameters(MethodHyperparameters):
    epsilon2: float
    conditional_independence_tester_class: type
    alpha: float = 0


def find_approx_screening_set(
    instance_data: ProblemInstance, 
    X: set,
    Y: set,
    S: set,
    tester: ConditionalIndependenceTester
):
    samples = instance_data.samples
    ND = instance_data.nondescendants

    num_nodes = samples.shape[1]
    for k in range(num_nodes):
        for Sprime in itr.combinations(ND, k):
            res1 = tester.test(Y, S - set(Sprime), X | set(Sprime))
            res2 = tester.test(X, set(Sprime) - S, S)
            if not res1["reject"] and not res2["reject"]:
                return set(Sprime)
            
    return ND


class Bamba(CausalEffectEstimationMethod):
    def __init__(self, hyperparameters: BambaHyperparameters):
        super().__init__(hyperparameters)
        self.hyperparameters = hyperparameters
    
    @property
    def name(self):
        return "Bamba"
    
    @property
    def key(self):
        return f"Bamba(alpha={self.hyperparameters.alpha}, epsilon2={self.hyperparameters.epsilon2})"
    
    @property
    def color(self):
        return "g"

    def estimate(
            self, 
            problem_instance: ProblemInstance, 
            query: Query, 
            combo_check: bool = False
        ) -> float:  
        samples = problem_instance.samples
        tester: ConditionalIndependenceTester = self.hyperparameters.conditional_independence_tester_class(
            samples, 
            self.hyperparameters.epsilon2,
            problem_instance.alphabet_sizes
        )

        X = {query.action}
        Y = {query.target}
        S = find_approx_markov_blanket(problem_instance, X, tester)
        Sprime = find_approx_screening_set(problem_instance, X, Y, S, tester)

        adjustment_used = Sprime
        if combo_check:
            n = len(samples)
            Z = problem_instance.nondescendants
            Sigma_X_sizes = np.array(problem_instance.alphabet_sizes)[list(X)]
            Sigma_S_sizes = np.array(problem_instance.alphabet_sizes)[list(S)]
            lb = estimate_alpha_lower_bound(
                problem_instance.samples,
                list(Sigma_X_sizes),
                list(Sigma_S_sizes),
                query,
                S
            )

            Sigma_Z_sizes = np.array(problem_instance.alphabet_sizes)[list(Z)]
            Sigma_X_size = np.prod(Sigma_X_sizes)
            Sigma_Z_size = np.prod(Sigma_Z_sizes)
            LHS = len(S) * np.sqrt(Sigma_X_size / Sigma_Z_size)
            RHS = max(Sigma_Z_size / n, max(lb / Sigma_Z_size, lb * lb))
            if LHS >= RHS:
                adjustment_used = Z
            else:
                print("Combi check passed: using", adjustment_used, "instead of", Z)

        return adjust_by_S(
            samples, 
            query, 
            adjustment_used, 
            problem_instance.alphabet_sizes, 
            alpha=self.hyperparameters.alpha
        )

            
# === IMPORTS: BUILT-IN ===
import itertools as itr
from time import time
from dataclasses import dataclass

# === IMPORTS: LOCAL ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters
from src.synthetic_generation import ProblemInstance
from src.ci_test.custom_ci_tester import ConditionalIndependenceTester
from src.utils import Query, adjust_by_S


@dataclass
class AmbaHyperparameters(MethodHyperparameters):
    epsilon2: float
    conditional_independence_tester_class: type
    alpha: float = 0


def find_approx_markov_blanket(
    instance_data: ProblemInstance, 
    nodes: set, 
    tester: ConditionalIndependenceTester,
):
    samples = instance_data.samples
    ND = instance_data.nondescendants

    num_nodes = samples.shape[1]
    for k in range(num_nodes):
        for S in itr.combinations(ND, k):
            S_complement = ND - set(S)
            res = tester.test(nodes, S_complement, set(S))
            if not res["reject"]:
                return set(S)
            
    print("Amba using entire ND")
    return ND


class Amba(CausalEffectEstimationMethod):
    def __init__(self, hyperparameters: AmbaHyperparameters):
        super().__init__(hyperparameters)
        self.hyperparameters = hyperparameters
    
    @property
    def name(self):
        return "Amba"
    
    @property
    def key(self):
        return f"Amba(alpha={self.hyperparameters.alpha}, epsilon2={self.hyperparameters.epsilon2})"
    
    @property
    def color(self):
        return "g"

    def estimate(self, problem_instance: ProblemInstance, query: Query) -> float:
        samples = problem_instance.samples
        tester: ConditionalIndependenceTester = self.hyperparameters.conditional_independence_tester_class(
            samples, 
            self.hyperparameters.epsilon2,
            problem_instance.alphabet_sizes
        )

        S = find_approx_markov_blanket(problem_instance, {query.action}, tester)
        return adjust_by_S(
            samples, 
            query, 
            S, 
            problem_instance.alphabet_sizes, 
            alpha=self.hyperparameters.alpha
        )

            
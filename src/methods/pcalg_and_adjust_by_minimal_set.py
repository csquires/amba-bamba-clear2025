# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass

# === IMPORTS: THIRD-PARTY ===
import causaldag as cd
from causaldag import pcalg
from causaldag import MemoizedCI_Tester
from graphical_models import DAG, PDAG

# === IMPORTS: LOCAL ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod, MethodHyperparameters
from src.synthetic_generation import ProblemInstance
from src.ci_test.plugin_ci_tester import PluginConditionalIndependenceTester
from src.utils import Query, adjust_by_S, get_minimal_adjset


@dataclass
class PcalgAndAdjustByMinimalSetHyperparameters(MethodHyperparameters):
    epsilon2: float
    alpha: float = 0


class PcalgAndAdjustByMinimalSet(CausalEffectEstimationMethod):
    @property
    def name(self):
        return "PC-Min"
    
    @property
    def color(self):
        return "purple"
    
    @property
    def key(self):
        return f"PC-Min(alpha={self.hyperparameters.alpha}, epsilon2={self.hyperparameters.epsilon2})"
    
    def estimate(self, problem_instance: ProblemInstance,  query: Query) -> float:
        # CREATE A WRAPPER THAT MAKES OUR CONDITIONAL INDEPENDENCE TESTER COMPATIBLE WITH CAUSALDAG
        def ci_test(samples, i, j, cond_set):
            tester = PluginConditionalIndependenceTester(
                samples, 
                self.hyperparameters.epsilon2,
                problem_instance.alphabet_sizes
            )
            return tester.test({i}, {j}, set(cond_set))
        
        nodes = list(range(problem_instance.dag.nnodes))
        
        # RUN PCALG
        samples = problem_instance.samples
        ci_tester = MemoizedCI_Tester(ci_test, samples)
        res = pcalg(nodes, ci_tester)
        assert len(res.nodes) == len(nodes)

        # Extract skeleton
        G = PDAG(nodes=res.nodes, arcs=set(), edges=res.skeleton)
        assert len(G.nodes) == len(nodes)

        # Add X to Y arc, even if PC failed to detect it
        X = query.action
        Y = query.target
        if frozenset({X, Y}) in G.edges:
            G.replace_edge_with_arc((X, Y))
        else:
            G._add_arc(X, Y)  # maybe don't add this edge??? This is more favorable to PC algorithm

        # Add non-descendent arcs
        Z = problem_instance.nondescendants
        for i, j in G.edges:
            if i == X and j in Z:
                G.replace_edge_with_arc((j, i))
            elif j == X and i in Z:
                G.replace_edge_with_arc((i, j))
            elif i == Y and j in Z:
                G.replace_edge_with_arc((j, i))
            elif j == Y and i in Z:
                G.replace_edge_with_arc((i, j))

        # Add back consistent v-structures
        for i, j in res.vstructs():
            if (j, i) not in G.arcs and frozenset({i, j}) in G.edges:
                G.replace_edge_with_arc((i, j))

        # Run Meek rules
        G.to_complete_pdag()

        # Pick an arbitrary DAG consistent with the CPDAG to compute adjustment sets
        # e.g. via "root picking"; see https://arxiv.org/pdf/2012.09679
        # Such an approach is valid via Lemma 10 of https://www.jmlr.org/papers/volume18/16-319/16-319.pdf
        for node in nodes:
            # Point all unoriented edges out of this node
            for nbr in G.neighbors_of(node):
                if frozenset({node, nbr}) in G.edges:
                    G.replace_edge_with_arc((node, nbr))

            # Run Meek rules
            G.to_complete_pdag()
        assert len(G.edges) == 0
        adj_set = get_minimal_adjset(G, X, Y)

        return adjust_by_S(
            samples,
            query,
            adj_set,
            problem_instance.alphabet_sizes,
            alpha=self.hyperparameters.alpha
        )
        

if __name__ == "__main__":
    import numpy as np

    nnodes = 8
    domain_size = 2
    query = Query(nnodes-2, nnodes-1, 0, 1)

    # CREATE DAG STRUCTURE
    dag = cd.rand.directed_erdos(nnodes-2, 1, random_order=False)
    arcs_to_action = {(0, nnodes-2)}
    arcs_to_target = {(i, nnodes-1) for i in range(nnodes-2)}
    dag.add_arcs_from(arcs_to_action)
    dag.add_arcs_from(arcs_to_target)
    dag.add_arc(nnodes-2, nnodes-1)

    # CREATE DAG MECHANISMS
    node_alphabets = {node: np.arange(domain_size) for node in range(nnodes)}
    # ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)
    ddag = cd.rand.rand_noisy_or_dag(dag, p_low=0.8)

    # COMPUTE TRUE CAUSAL EFFECT
    # true_effect = compute_causal_effects(ddag, query)

    samples = ddag.sample(50)
    instance_data = ProblemInstance(ddag, domain_size, samples)

    hyperparameters = PcalgAndAdjustByMinimalSetHyperparameters(0.1, 0.1)
    method = PcalgAndAdjustByMinimalSet(hyperparameters)
    method.estimate(instance_data, query)




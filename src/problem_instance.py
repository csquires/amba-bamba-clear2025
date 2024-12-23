# === IMPORTS: BUILT-IN ===
from typing import NewType, Dict, Union

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd

# === IMPORTS: LOCAL ===
from src.utils import compute_causal_effects
from src.custom_types import Query


class ProblemInstance:
    def __init__(
        self,
        dag: cd.DiscreteDAG,
        observed_graph: Union[cd.DAG, cd.AncestralGraph],
        alphabet_sizes: int,
        samples: np.ndarray,
        query: Query
    ):
        self.dag = dag
        self.observed_graph = observed_graph
        self.alphabet_sizes = alphabet_sizes
        self.samples = samples
        self.query = query

        # ADDITIONAL DATA FROM THE DAG
        self.nondescendants = set(range(observed_graph.nnodes)) - set(observed_graph.descendants_of(query.action)) - {query.action}
        self.parents = self.observed_graph.parents_of(query.action)
        
        if isinstance(self.observed_graph, cd.AncestralGraph):
            self.spouses = self.observed_graph.spouses_of(query.action)
        else:
            self.spouses = set()

        # TRUE EFFECT
        self.true_effect = compute_causal_effects(self.dag, query)


# an InstanceDict maps (nsamples, dag_index, run_index) to a ProblemInstance object
InstanceKey = NewType("InstanceKey", tuple[str, int, int])
InstanceDict = NewType("InstanceDict", Dict[InstanceKey, ProblemInstance])
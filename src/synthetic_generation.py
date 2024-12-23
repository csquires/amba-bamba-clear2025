# === IMPORTS: BUILT-IN ===
import os
import random
import itertools as itr
from typing import List
from dataclasses import dataclass
import pickle
from copy import deepcopy

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd

# === IMPORTS: LOCAL ===
from src.custom_types import Query
from src.problem_instance import ProblemInstance, InstanceDict
from src.paths import SYNTHETIC_DATA_DIR


@dataclass
class SyntheticDataConfig:
    nsamples_list: List[int]
    nnodes: int
    density: float = 0.5
    domain_size: int = 2

    ndags: int = 50
    nruns_per_dag: int = 10
    seed: int = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(0, 2**20)

        self.identifier = f"nnodes={self.nnodes},density={self.density},seed={self.seed}"

    def copy(self):
        return deepcopy(self)



def generate_structure(
    config: SyntheticDataConfig,
    nnodes: int,
    method: str = "erdos"
) -> cd.DAG:
    cfg = config
    
    if method == "erdos":
        dag = cd.rand.directed_erdos(nnodes, cfg.density, random_order=False)
        dag.add_arc(nnodes-2, nnodes-1)
    elif method == "semi-erdos":
        dag = cd.rand.directed_erdos(nnodes-2, cfg.density, random_order=False)
        nd = set(range(nnodes-2))
        common_parents = set(random.choices(list(nd), k=2))
        X_parents = set(random.choices(list(nd - common_parents), k=1))
        Y_parents = set(random.choices(list(nd - common_parents - X_parents), k=2))
        dag.add_arcs_from({(i, nnodes-2) for i in common_parents | X_parents})
        dag.add_arcs_from({(i, nnodes-1) for i in common_parents | Y_parents})
        dag.add_arc(nnodes-2, nnodes-1)
    elif method == "semi-erdos-2":
        dag = cd.rand.directed_erdos(nnodes-2, cfg.density, random_order=False)
        X_parents = sorted(np.random.choice(nnodes-2, 1 + np.random.choice(3)))
        Y_parents = sorted(np.random.choice(nnodes-2, 1 + np.random.choice(3)))
        dag.add_arcs_from({(i, nnodes-2) for i in X_parents})
        dag.add_arcs_from({(i, nnodes-1) for i in Y_parents})
        dag.add_arc(nnodes-2, nnodes-1)
    elif method == "small-screening-set":
        grandparents = set(range((nnodes - 2) // 3))
        parents = set(range(nnodes - 2)) - grandparents
        dag = cd.DAG()
        dag.add_arcs_from({(i, j) for i in grandparents for j in parents})
        dag.add_arcs_from({(i, nnodes-2) for i in parents})
        dag.add_arcs_from({(i, nnodes-1) for i in grandparents})
        dag.add_arc(nnodes-2, nnodes-1)

    return dag


class SyntheticDataGenerator:
    def __init__(self, config: SyntheticDataConfig, overwrite=False):
        self.config = config
        self.overwrite = overwrite

    @property
    def save_file(self):
        cfg = self.config
        return f"{SYNTHETIC_DATA_DIR}/{cfg.identifier}.pkl"

    def generate_dags(self) -> List[cd.DiscreteDAG]:
        cfg = self.config

        dags = []
        for _ in range(cfg.ndags):
            # === Generate the structure
            dag = generate_structure(cfg, cfg.nnodes, method="small-screening-set")

            # === Generate the mechanisms
            node_alphabets = {node: np.arange(cfg.domain_size) for node in range(cfg.nnodes)}
            ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)

            # === Tilt the distribution of Y
            y_ix = cfg.nnodes-1
            tilt = 2
            y_conditional = ddag.conditionals[y_ix]
            y1_cond = y_conditional[..., 1]
            y0_cond = y_conditional[..., 0]
            new_y1_cond_unnorm = y1_cond * np.exp(tilt)
            new_y1_cond = new_y1_cond_unnorm / (new_y1_cond_unnorm + y0_cond)
            new_y0_cond = y0_cond / (new_y1_cond_unnorm + y0_cond)
            new_y_cond = np.stack([new_y0_cond, new_y1_cond], axis=-1)
            
            ddag.set_conditional(y_ix, new_y_cond)

            # === Add to list
            dags.append(ddag)

        return dags
    
    def generate_problem_instances(self) -> InstanceDict:
        cfg = self.config
        dags = self.generate_dags()

        instances = dict()
        for nsamples, d_ix, r_ix in itr.product(cfg.nsamples_list, range(cfg.ndags), range(cfg.nruns_per_dag)):
            # === Create the problem instance
            samples = dags[d_ix].sample(nsamples)
            alphabet_sizes = [cfg.domain_size] * cfg.nnodes
            query = Query(cfg.nnodes-2, cfg.nnodes-1, 0, 1)
            problem_instance = ProblemInstance(
                dags[d_ix], 
                dags[d_ix], 
                alphabet_sizes, 
                samples, 
                query
            )

            # === Add the instance to the dictionary
            instance_key = (nsamples, d_ix, r_ix)
            instances[instance_key] = problem_instance

        os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
        pickle.dump(instances, open(self.save_file, "wb"))
        return instances

# === IMPORTS: BUILT-IN ===
import os
import itertools as itr
from typing import List, Union
from dataclasses import dataclass
import pickle
from copy import deepcopy

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd

# === IMPORTS: LOCAL ===
from src.custom_types import Query
from src.paths import SEMISYNTHETIC_DATA_DIR
from src.problem_instance import ProblemInstance, InstanceDict



@dataclass
class SemisyntheticDataConfig:
    nsamples_list: List[int]
    ddag: cd.DiscreteDAG
    query: Query
    name: str
    nruns_per_dag: int = 10
    observed_variable_ixs: List[int] = None
    seed: int = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(0, 2**20)

        self.identifier = f"name={self.name},seed={self.seed}"

        if self.observed_variable_ixs is None:
            self.observed_variable_ixs = list(range(self.ddag.nnodes))
            self.observed_graph = self.ddag
        else:
            latent_nodes = set(range(self.ddag.nnodes)) - set(self.observed_variable_ixs)
            self.observed_graph = self.ddag.marginal_mag(latent_nodes)

    def copy(self):
        return deepcopy(self)
    


class SemisyntheticDataGenerator:
    def __init__(self, config: SemisyntheticDataConfig, overwrite=False):
        self.config = config
        self.overwrite = overwrite

    @property
    def save_file(self):
        cfg = self.config
        return f"{SEMISYNTHETIC_DATA_DIR}/{cfg.identifier}.pkl"

    def generate_problem_instances(self) -> InstanceDict:
        cfg = self.config

        instances = dict()
        for nsamples, d_ix, r_ix in itr.product(cfg.nsamples_list, range(1), range(cfg.nruns_per_dag)):
            # === Create the problem instance
            samples = cfg.ddag.sample(nsamples)
            alphabet_sizes = [len(cfg.ddag.node_alphabets[i]) for i in range(cfg.ddag.nnodes)]
            problem_instance = ProblemInstance(
                cfg.ddag, 
                cfg.observed_graph, 
                alphabet_sizes, 
                samples, 
                cfg.query
            )

            # === Add the instance to the dictionary
            instance_key = (nsamples, d_ix, r_ix)
            instances[instance_key] = problem_instance

        os.makedirs(SEMISYNTHETIC_DATA_DIR, exist_ok=True)
        pickle.dump(instances, open(self.save_file, "wb"))
        return instances
# === IMPORTS: BUILT-IN ===
from typing import List, NewType
import itertools as itr
from collections import defaultdict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: THIRD-PARTY ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod
from src.synthetic_generation import SyntheticDataConfig
from src.method_runner import ResultDict

# === An AvgResultKey is a tuple of the form (method_name, nsamples, dag_index) ===
AvgResultKey = NewType("InstanceKey", tuple[str, int, int])


class ResultComputer:
    def __init__(self, data_config: SyntheticDataConfig, overwrite=False):
        self.data_config = data_config
        self.overwrite = overwrite

    def compute_mse_per_dag(
        self, 
        results: ResultDict, 
        methods: List[CausalEffectEstimationMethod]
    ):
        cfg = self.data_config

        mse_per_dag = defaultdict(list)
        for method, nsamples, d_ix in itr.product(methods, cfg.nsamples_list, range(cfg.ndags)):
            squared_errors = np.zeros(cfg.nruns_per_dag)
            for r_ix in range(cfg.nruns_per_dag):
                result_key = (method.key, nsamples, d_ix, r_ix)
                squared_error = results[result_key].squared_error
                squared_errors[r_ix] = squared_error
                
            key = (method.key, nsamples)
            mse_per_dag[key].append(np.mean(squared_errors))
        
        return dict(mse_per_dag)

    def get_normalized_errors(
        self, 
        results: ResultDict, 
        methods: List[CausalEffectEstimationMethod],
        baseline_method="ND-Adjust"
    ):
        cfg = self.data_config
        mse_per_dag = self.compute_mse_per_dag(results, methods)

        baseline_errors = dict()
        for nsamples in cfg.nsamples_list:
            baseline_errors[nsamples] = np.array([
                mse_per_dag[(baseline_method, nsamples)][d_ix]
                for d_ix in range(cfg.ndags)
            ])

        normalized_error_dict = dict()
        for method in methods:
            for nsamples in cfg.nsamples_list:
                errors = np.array([
                    mse_per_dag[(method.key, nsamples)][d_ix]
                    for d_ix in range(cfg.ndags)
                ])
                normalized_errors = errors / baseline_errors[nsamples]
                normalized_error_dict[(nsamples, method.key)] = normalized_errors

        return normalized_error_dict
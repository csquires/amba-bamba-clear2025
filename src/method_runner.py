# === IMPORTS: BUILT-IN ===
import os
import pickle
from time import time
from typing import List, Dict, NewType
from dataclasses import dataclass
import itertools as itr
from multiprocessing import Pool

# === IMPORTS: THIRD-PARTY ===
from tqdm import trange, tqdm

# === IMPORTS: THIRD-PARTY ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod
from src.problem_instance import ProblemInstance
from src.custom_types import Query
from src.synthetic_generation import SyntheticDataConfig, InstanceDict
from src.paths import SYNTHETIC_RESULT_DIR


@dataclass
class ResultData:
    problem_instance: ProblemInstance
    estimated_effect: float
    squared_error: float
    time_spent: float


# a ResultDict maps (method.key, nsamples, dag_index, run_index) to a ResultData object
ResultKey = NewType("ResultKey", tuple[str, int, int, int])
ResultDict = NewType("ResultDict", Dict[ResultKey, ResultData])


class MethodRunner:
    def __init__(self, data_config: SyntheticDataConfig):
        self.data_config = data_config

    @property
    def result_filename(self):
        cfg = self.data_config
        os.makedirs(SYNTHETIC_RESULT_DIR, exist_ok=True)
        filename = f"{SYNTHETIC_RESULT_DIR}/{cfg.identifier}.pkl"
        return filename

    def run_instance(
        self, 
        method: CausalEffectEstimationMethod, 
        problem_instance: ProblemInstance
    ):
        # GENERATE SAMPLES AND COMPUTE ESTIMATED CAUSAL EFFECT
        start = time()
        estimated_effect = method.estimate(problem_instance, problem_instance.query)
        time_spent = time() - start
        
        # COLLECT RESULTS
        result = ResultData(
            problem_instance=problem_instance,
            estimated_effect=estimated_effect,
            squared_error=(problem_instance.true_effect - estimated_effect) ** 2,
            time_spent=time_spent
        )

        return result
    
    def run_method(
        self,
        method: CausalEffectEstimationMethod,
        instances: InstanceDict,
    ):
        method_results = dict()
        for instance_key, problem_instance in tqdm(instances.items()):
            result_key = (method.key, *instance_key)
            result = self.run_instance(method, problem_instance)
            method_results[result_key] = result

        # ==================== OLD, TO BE DEPRECATED ====================
        # for d_ix in trange(cfg.ndags):
        #     for r_ix in range(cfg.nruns_per_dag):
        #         instance_key = (nsamples, d_ix, r_ix)
        #         problem_instance = instances[instance_key]

        #         # === Run `method` on the problem instance
        #         result_key = (method.key, nsamples, d_ix, r_ix)
        #         result = self.run_instance(method, problem_instance)
        #         method_results[result_key] = result  
        # ==================================================================

        return method_results
        

    def run(
        self, 
        instances: InstanceDict, 
        methods: List[CausalEffectEstimationMethod], 
        overwrite=False,
        parallelize: bool = False,
        n_processes: int = None
    ) -> ResultDict:
        cfg = self.data_config

        # CHECK IF RESULT FILE ALREADY EXISTS
        if os.path.exists(self.result_filename) and not overwrite:
            results = self.retrieve_results()
        else:
            results = dict()

        if not parallelize:
            for method in methods:
                method_results = self.run_method(method, instances)
                results |= method_results
        else:
            n_processes = n_processes if n_processes is not None else os.cpu_count() - 1
            with Pool(n_processes) as pool:
                method_results_list = pool.starmap(
                    self.run_method,
                    [(method, nsamples, instances) for nsamples, method in itr.product(cfg.nsamples_list, methods)]
                )
            results = dict(res for method_results in method_results_list for res in method_results.items())
            
        pickle.dump(results, open(self.result_filename, "wb"))
        return results

    def retrieve_results(self) -> ResultDict:
        if not os.path.exists(self.result_filename):
            raise ValueError("The results do not exist. Please run().")

        results = pickle.load(open(self.result_filename, "rb"))
        return results
    

    
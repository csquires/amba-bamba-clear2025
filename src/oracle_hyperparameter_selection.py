# === IMPORTS: BUILT-IN ===
from typing import List, Dict

# === IMPORTS: THIRD-PARTY ===
import pandas as pd

# === IMPORTS: LOCAL ===
from src.result_computer import ResultComputer
from src.method_runner import MethodRunner
from src.synthetic_generation import SyntheticDataConfig, SyntheticDataGenerator
from src.semisynthetic_generation import SemisyntheticDataConfig, SemisyntheticDataGenerator


class OracleHyperparameterSelection:
    def __init__(
        self, 
        data_config: SyntheticDataConfig,
    ):
        self.data_config = data_config

    def select_hyperparameters(
        self, 
        method_class, 
        hyperparameter_grid: List[Dict],
        parallelize: bool = False,
        n_processes: int = None
    ):
        # ==============================================================================
        # STEP 1: CREATE SYNTHETIC DATA
        generator = SyntheticDataGenerator(self.data_config, overwrite=False)
        instances = generator.generate_problem_instances()

        # ==============================================================================
        # STEP 2: RUN METHODS
        methods = [
            method_class(hyperparameters) 
            for hyperparameters in hyperparameter_grid
        ]
        method_runner = MethodRunner(self.data_config)
        results = method_runner.run(
            instances, 
            methods, 
            overwrite=True, 
            parallelize=parallelize, 
            n_processes=n_processes
        )

        # ==============================================================================
        # STEP 3: COMPUTE RESULTS
        result_computer = ResultComputer(self.data_config)
        mses = result_computer.compute_mse_per_dag(results, methods)
        mse_df = pd.DataFrame.from_dict(mses, orient="index")

        # ==============================================================================
        # STEP 4: PICK BEST HYPERPARAMETERS
        indicator_df = mse_df.apply(lambda x: (x == x.min()).astype(int), axis=0)
        percent_best = indicator_df.mean(axis=1)
        best_ix = percent_best.idxmax()
        return hyperparameter_grid[percent_best.index.to_list().index(best_ix)]
    

class OracleHyperparameterSelectionSemisynthetic:
    def __init__(
        self, 
        data_config: SemisyntheticDataConfig,
    ):
        self.data_config = data_config

    def select_hyperparameters(
        self, 
        method_class, 
        hyperparameter_grid: List[Dict],
        parallelize: bool = False,
        n_processes: int = None
    ):
        # ==============================================================================
        # STEP 1: CREATE SYNTHETIC DATA
        generator = SemisyntheticDataGenerator(self.data_config, overwrite=False)
        instances = generator.generate_problem_instances()

        # ==============================================================================
        # STEP 2: RUN METHODS
        methods = [
            method_class(hyperparameters) 
            for hyperparameters in hyperparameter_grid
        ]
        method_runner = MethodRunner(self.data_config)
        results = method_runner.run(
            instances, 
            methods, 
            overwrite=True, 
            parallelize=parallelize, 
            n_processes=n_processes
        )

        # ==============================================================================
        # STEP 3: COMPUTE RESULTS
        result_computer = ResultComputer(self.data_config)
        mses = result_computer.compute_mse_per_dag(results, methods)
        mse_df = pd.DataFrame.from_dict(mses, orient="index")

        # ==============================================================================
        # STEP 4: PICK BEST HYPERPARAMETERS
        indicator_df = mse_df.apply(lambda x: (x == x.min()).astype(int), axis=0)
        percent_best = indicator_df.mean(axis=1)
        best_ix = percent_best.idxmax()
        return hyperparameter_grid[percent_best.index.to_list().index(best_ix)]


if __name__ == "__main__":
    from methods.adjust_by_markov_blanket import AdjustByMarkovBlanket, AdjustByMarkovBlanketHyperparameters

    data_config = SyntheticDataConfig(
        nsamples_list=[1000],
        nnodes=10,
        density=0.5,
        domain_size=2,
        ndags=5,
        nruns_per_dag=2,
        seed=1000
    )
    hp_selector = OracleHyperparameterSelection(data_config)
    hyperparameter_grid = [
        AdjustByMarkovBlanketHyperparameters(alpha=0.1),
        AdjustByMarkovBlanketHyperparameters(alpha=0.2),
        AdjustByMarkovBlanketHyperparameters(alpha=0.3),
        AdjustByMarkovBlanketHyperparameters(alpha=0.4),
        AdjustByMarkovBlanketHyperparameters(alpha=0.5),
    ]
    errors = hp_selector.select_hyperparameters(
        AdjustByMarkovBlanket, 
        hyperparameter_grid,
        parallelize=True
    )
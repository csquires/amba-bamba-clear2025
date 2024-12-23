# === IMPORTS: BUILT-IN ===
import itertools as itr
import pickle

# === IMPORTS: LOCAL (CLASSES) ===
from src.synthetic_generation import SyntheticDataConfig, SyntheticDataGenerator
from src.method_runner import MethodRunner
from src.oracle_hyperparameter_selection import OracleHyperparameterSelection

# === IMPORTS: LOCAL (METHODS) ===
from src.methods.adjust_by_full_set import AdjustByFullSet, AdjustByFullSetHyperparameters
from src.methods.adjust_by_markov_blanket import AdjustByMarkovBlanket, AdjustByMarkovBlanketHyperparameters
from src.methods.adjust_by_minimal_set import AdjustByMinimalSet, AdjustByMinimalSetHyperparameters
from src.methods.pcalg_and_adjust_by_minimal_set import PcalgAndAdjustByMinimalSet, PcalgAndAdjustByMinimalSetHyperparameters

from src.methods.amba import Amba, AmbaHyperparameters
from src.methods.bamba import Bamba, BambaHyperparameters
from src.methods.combination import Combination, CombinationHyperparameters
from src.ci_test.plugin_ci_tester import PluginConditionalIndependenceTester

PARALLELIZE = False


# ==============================================================================
# STEP 1: CREATE SYNTHETIC DATA
training_data_config = SyntheticDataConfig(
    nsamples_list=[500],
    nnodes=10,
    density=0.5,
    domain_size=2,
    ndags=10,
    nruns_per_dag=5,
    seed=0
)
test_data_config = training_data_config.copy()
test_data_config.seed = 3

training_generator = SyntheticDataGenerator(training_data_config, overwrite=False)
test_generator = SyntheticDataGenerator(test_data_config, overwrite=False)
training_instances = training_generator.generate_problem_instances()
test_instances = test_generator.generate_problem_instances()

# ==============================================================================
# STEP 2: DEFINE HYPERPARAMETER GRIDS
alpha_grid = [0.1, 0.25, 0.5, 1, 1.5, 2]
epsilon2_grid = [0.05, 0.075, 0.1, 0.2]
epsilon2_grid_pc = [0.01, 0.05]

# Oracle methods
full_adj_grid = [AdjustByFullSetHyperparameters(alpha=alpha) for alpha in alpha_grid]
mb_adj_grid = [AdjustByMarkovBlanketHyperparameters(alpha=alpha) for alpha in alpha_grid]
minimal_adj_grid = [AdjustByMinimalSetHyperparameters(alpha=alpha) for alpha in alpha_grid]

# Non-oracle methods
pcalg_and_minimal_adj_grid = [
    PcalgAndAdjustByMinimalSetHyperparameters(
        alpha=alpha,
        epsilon2=epsilon2
    )
    for alpha, epsilon2 in itr.product(alpha_grid, epsilon2_grid_pc)
]
amba_grid = [
    AmbaHyperparameters(
        alpha=alpha,
        conditional_independence_tester_class=PluginConditionalIndependenceTester,
        epsilon2=epsilon2
    )
    for alpha, epsilon2 in itr.product(alpha_grid, epsilon2_grid)
]
bamba_grid = [
    BambaHyperparameters(
        alpha=alpha,
        conditional_independence_tester_class=PluginConditionalIndependenceTester,
        epsilon2=epsilon2
    )
    for alpha, epsilon2 in itr.product(alpha_grid, epsilon2_grid)
]
combination_grid = [
    CombinationHyperparameters(
        alpha=alpha,
        conditional_independence_tester_class=PluginConditionalIndependenceTester,
        epsilon2=epsilon2
    )
    for alpha, epsilon2 in itr.product(alpha_grid, epsilon2_grid)
]

# ==============================================================================
# STEP 3: SELECT ORACLE HYPERPARAMETERS ON TRAINING DATA
hyperparameter_selector = OracleHyperparameterSelection(training_data_config)

# Oracle methods
full_adj_params = hyperparameter_selector.select_hyperparameters(AdjustByFullSet, full_adj_grid, parallelize=PARALLELIZE)
print("Selected hyperparameters for Z-Adjust:", full_adj_params)
mb_adj_params = hyperparameter_selector.select_hyperparameters(AdjustByMarkovBlanket, mb_adj_grid, parallelize=PARALLELIZE)
print("Selected hyperparameters for MB-Adjust:", mb_adj_params)
minimal_adj_params = hyperparameter_selector.select_hyperparameters(AdjustByMinimalSet, minimal_adj_grid, parallelize=PARALLELIZE)
print("Selected hyperparameters for Min-Adjust:", minimal_adj_params)

# Non-oracle methods
# pcalg_and_minimal_adj_params = hyperparameter_selector.select_hyperparameters(PcalgAndAdjustByMinimalSet, pcalg_and_minimal_adj_grid, parallelize=PARALLELIZE)
# print("Selected hyperparameters for Min-Adjust:", pcalg_and_minimal_adj_params)
amba_params = hyperparameter_selector.select_hyperparameters(Amba, amba_grid, parallelize=PARALLELIZE)
print("Selected hyperparameters for AMBA:", amba_params)
bamba_params = hyperparameter_selector.select_hyperparameters(Bamba, bamba_grid, parallelize=PARALLELIZE)
print("Selected hyperparameters for BAMBA:", bamba_params)
# combination_params = hyperparameter_selector.select_hyperparameters(Combination, combination_grid, parallelize=PARALLELIZE)
# print("Selected hyperparameters for Combination:", combination_params)

# ==============================================================================
# STEP 4: RUN METHODS ON TEST DATA USING ORACLE HYPERPARAMETERS 

# pa_adj_params = AdjustByParentsHyperparameters(alpha=0.1)
# nd_adj_params = AdjustByAllNonDescendantsHyperparameters(alpha=0.1)
# amba_params = AmbaHyperparameters(alpha=0.1, epsilon2=0.2, conditional_independence_tester_class=PluginConditionalIndependenceTester)
# bamba_params = BambaHyperparameters(alpha=0.1, epsilon2=0.2, conditional_independence_tester_class=PluginConditionalIndependenceTester)
methods = [
    AdjustByFullSet(full_adj_params),
    AdjustByMarkovBlanket(mb_adj_params),
    AdjustByMinimalSet(minimal_adj_params),
    # PcalgAndAdjustByMinimalSet(pcalg_and_minimal_adj_params),
    Amba(amba_params),
    Bamba(bamba_params),
    # Combination(combination_params),
]

# run methods
method_runner = MethodRunner(test_data_config)
results = method_runner.run(test_instances, methods, overwrite=True)
pickle.dump(methods, open("experiments/synthetic/methods.pkl", "wb"))
pickle.dump(results, open("experiments/synthetic/results.pkl", "wb"))

# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: LOCAL (CLASSES) ===
from src.synthetic_generation import SyntheticDataConfig
from src.result_computer import ResultComputer
from src.result_plotter import ResultPlotter


# ==============================================================================
# STEP 1: RETRIEVE RESULTS
test_data_config = SyntheticDataConfig(
    nsamples_list=[500],
    nnodes=10,
    density=0.5,
    domain_size=2,
    ndags=10,
    nruns_per_dag=5,
    seed=3
)
methods = pickle.load(open("experiments/synthetic/methods.pkl", "rb"))
results = pickle.load(open("experiments/synthetic/results.pkl", "rb"))

# ==============================================================================
# STEP 2: COMPUTE RESULTS
result_computer = ResultComputer(test_data_config)
baseline = methods[0].key  # ND-Adjust
normalized_errors = result_computer.get_normalized_errors(results, methods, baseline_method=baseline)

# ==============================================================================
# STEP 3: PLOT RESULTS
result_plotter = ResultPlotter(test_data_config, normalized_errors)
result_plotter.boxplot(methods, normalized_errors)
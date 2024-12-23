# === IMPORTS: BUILT-IN ===
import os
from typing import List

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# === IMPORTS: THIRD-PARTY ===
from src.methods.causal_effect_estimation_method import CausalEffectEstimationMethod
from src.synthetic_generation import SyntheticDataConfig


class ResultPlotter:
    def __init__(self, data_config: SyntheticDataConfig, overwrite=False):
        self.data_config = data_config
        self.overwrite = overwrite

    @property
    def figure_filename(self):
        cfg = self.data_config
        folder = f"figures/density={cfg.density}"
        filename = f"{folder}/seed={cfg.seed}.png"
        os.makedirs(folder, exist_ok=True)
        return filename
    
    @property
    def normalized_figure_filename(self):
        cfg = self.data_config
        folder = f"figures/density={cfg.density}"
        filename = f"{folder}/normalized_seed={cfg.seed}.png"
        os.makedirs(folder, exist_ok=True)
        return filename
    
    def boxplot(self, methods: List[CausalEffectEstimationMethod], normalized_error_dict):
        cfg = self.data_config

        plt.clf()
        sns.set_theme()
        plt.figure(figsize=(12, 6))

        method_names = [method.name for method in methods]
        nsamples = cfg.nsamples_list[0]
        values = np.concatenate([normalized_error_dict[(nsamples, method.key)] for method in methods])
        data = pd.DataFrame({
            'Category': np.repeat(method_names, cfg.ndags),
            'Value': values
        })

        # Create the box plot
        g = sns.boxplot(
            x='Category', 
            y='Value', 
            hue='Category', 
            data=data, 
            showfliers=False, 
            dodge=False,
            width=0.5
        )
        # g.legend_.remove()

        plt.tick_params(labelsize=18)
        plt.ylabel("Normalized error", fontsize=24)
        plt.xlabel("")
        plt.tight_layout()
        plt.plot()
        plt.savefig(self.normalized_figure_filename)

        print(f"Saved boxplot to {self.normalized_figure_filename}")

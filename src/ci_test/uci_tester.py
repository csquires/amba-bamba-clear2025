# === IMPORTS: BUILT-IN ===
from collections import Counter

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.stats.contingency import crosstab
from scipy.stats import random_table

# === IMPORTS: LOCAL ===
from src.ci_test.custom_ci_tester import ConditionalIndependenceTester
from src.utils import vector2scalar


class UCITester(ConditionalIndependenceTester):
    def Ustat2(self, sigma_qr: np.ndarray, sigma: int):
        sigma_q = sigma_qr.sum(axis=1)
        sigma_r = sigma_qr.sum(axis=0)
        A1 = np.sum(sigma_qr ** 2 - sigma_qr)
        A2 = np.sum(sigma_q ** 2 - sigma_q) * np.sum(sigma_r ** 2 - sigma_r)
        A3_terms = sigma_qr * (np.outer(sigma_q, sigma_r) - sigma_q[:, None] - sigma_r + 1)
        A3 = np.sum(A3_terms)
        # print(f"A1: {A1}")
        # print(f"A2: {A2}")
        # print(f"A3: {A3}")
        
        A2_coef = 1 / (sigma - 1) / (sigma - 2)
        A3_coef = 2 / (sigma - 2)
        return (A1 + A2_coef * A2 - A3_coef * A3) / sigma / (sigma - 3)
    
    def compute_statistic(self, contingency_table: np.ndarray, C_levels: list):
        T = 0
        for m in C_levels:
            count = contingency_table[m].sum()
            if count >= 4:
                T_m = self.Ustat2(contingency_table[m], count)
                T += count * T_m

        return T

    def test(self, A: set, B: set, C: set, npermutations=100) -> bool:
        # CONVERT VECTORS TO SCALARS (e.g., [0, 0, 0] ... [1, 1, 1] -> 0 ... 7)
        scalar_data = np.vstack((
            vector2scalar(self.data[:, list(C)]), 
            vector2scalar(self.data[:, list(A)]), 
            vector2scalar(self.data[:, list(B)])
        )).T

        # COMPUTE CONTINGENCY TABLE
        C_levels = np.unique(scalar_data[:, 0])
        A_levels = np.unique(scalar_data[:, 1])
        B_levels = np.unique(scalar_data[:, 2])
        crosstab_result = crosstab(
            scalar_data[:, 0], 
            scalar_data[:, 1], 
            scalar_data[:, 2],
            levels=[C_levels, A_levels, B_levels]
        )
        contingency_table = crosstab_result.count

        # COMPUTE STATISTIC FOR REAL DATA
        statistic = self.compute_statistic(contingency_table, C_levels)

        # CREATE PERMUTED VERSIONS OF THE CONTINGENCY TABLES
        permuted_contingency_tables = np.zeros((npermutations, *contingency_table.shape))
        for m in C_levels:
            row_sums = contingency_table[m].sum(axis=1)
            col_sums = contingency_table[m].sum(axis=0)
            permuted_subtable = random_table(row_sums, col_sums)
            permuted_contingency_tables[:, m] = permuted_subtable.rvs(npermutations)
            
        # COMPUTE STATISTICS FOR PERMUTED DATA
        permutation_statistics = np.empty(npermutations)
        for p_ix in range(npermutations):
            permuted_contingency_table = permuted_contingency_tables[p_ix]
            permutation_statistics[p_ix] = self.compute_statistic(permuted_contingency_table, C_levels)
        
        pval = np.mean(statistic >= permutation_statistics)
        res = dict(
            statistic=statistic,
            pval=pval,
            reject=pval >= self.epsilon
        )
        return res


if __name__ == "__main__":
    domain_size = 2
    dag = cd.DAG(arcs={(0, 1), (1, 2)})

    node_alphabets = {node: list(range(domain_size)) for node in dag.nodes}
    ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)

    samples = ddag.sample(100)
    tester = UCITester(samples, 0.05, domain_size)

    A = {0}
    B = {1}
    C = {2}

    res = tester.test(A, B, C)

    # ddag_est = cd.DiscreteDAG.fit(dag, samples, node_alphabets)
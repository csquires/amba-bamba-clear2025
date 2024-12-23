# === IMPORTS: BUILT-IN ===
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.stats.contingency import crosstab

# === IMPORTS: LOCAL ===
from src.ci_test.custom_ci_tester import ConditionalIndependenceTester
from src.utils import empirical_estimate, vector2scalar


class PluginConditionalIndependenceTesterOld(ConditionalIndependenceTester):
    def new_test(self, A: set, B: set, C: set) -> dict:
        # 0 = C, 1 = A, 2 = B
        full_dag = cd.DAG(arcs={(0, 1), (0, 2), (1, 2)})
        ci_dag = cd.DAG(arcs={(0, 1), (0, 2)})  # Missing A -> B

        # FIT DAGS TO DATA
        small_data = np.vstack((
            vector2scalar(self.data[:, list(C)]),
            vector2scalar(self.data[:, list(A)]),
            vector2scalar(self.data[:, list(B)]),
        )).T
        node_alphabets = {
            0: list(range(self.domain_size ** len(C))),
            1: list(range(self.domain_size ** len(A))),
            2: list(range(self.domain_size ** len(B)))
        }
        full_ddag = cd.DiscreteDAG.fit(full_dag, small_data, node_alphabets=node_alphabets)
        ci_ddag = cd.DiscreteDAG.fit(ci_dag, small_data, node_alphabets=node_alphabets)

        # COMPUTE TV DISTANCE
        full_marginal = full_ddag.get_marginals([0, 1, 2])
        ci_marginal = ci_ddag.get_marginals([0, 1, 2])
        # TODO: check if this is consistent
        # ci_marginal = full_ddag.get_marginals([0]) * full_ddag.get_conditional([1], [0]) * full_ddag.get_conditional([2], [0])
        statistic = np.sum(np.abs(full_marginal - ci_marginal))
        
        # RETURN RESULTS OF THE TEST
        res = dict(
            statistic=statistic,
            reject=statistic >= self.epsilon,
        )
        return res

    def test(self, A: set, B: set, C: set) -> dict:
        # RE-LABEL NODES
        C_ = set(range(len(C)))
        A_ = set(range(len(C), len(C) + len(A)))
        B_ = set(range(len(C) + len(A), len(C) + len(A) + len(B)))

        # CREATE TWO DAGS
        arcs_in_C = set(itr.combinations(C_, 2))
        arcs_in_A = set(itr.combinations(A_, 2))
        arcs_in_B = set(itr.combinations(B_, 2))
        arcs_C_to_A = set(itr.product(C_, A_))
        arcs_C_to_B = set(itr.product(C_, B_))
        arcs_A_to_B = set(itr.product(A_, B_))
        full_dag = cd.DAG(nodes=C_ | A_ | B_, arcs=arcs_in_C | arcs_in_A | arcs_in_B | arcs_C_to_A | arcs_C_to_B | arcs_A_to_B)
        ci_dag = cd.DAG(nodes=C_ | A_ | B_, arcs=arcs_in_C | arcs_in_A | arcs_in_B | arcs_C_to_A | arcs_C_to_B)

        # FIT DAGS TO DATA
        small_data = self.data[:, list(A | B | C)]
        node_alphabets = {node: range(self.domain_size) for node in range(self.data.shape[1])}
        full_ddag = cd.DiscreteDAG.fit(full_dag, small_data, node_alphabets=node_alphabets)
        ci_ddag = cd.DiscreteDAG.fit(ci_dag, small_data, node_alphabets=node_alphabets)

        # COMPUTE TV DISTANCE
        full_marginal = full_ddag.get_marginals(list(C_ | A_ | B_))
        ci_marginal = ci_ddag.get_marginals(list(C_ | A_ | B_))
        statistic = np.sum(np.abs(full_marginal - ci_marginal))
        
        # RETURN RESULTS OF THE TEST
        res = dict(
            statistic=statistic,
            reject=statistic >= self.epsilon,
        )
        return res

    """
    Returns whether sum_{a,b,c} P(c) * | P(a,b|c) - P(a|c) * P(b|c) | <= eps
    """
    def old_test(self, A: set, B: set, C: set) -> bool:
        # === REPLICATE DEFINITION 3 ===
        # Estimate P(C)
        P_C = empirical_estimate(self.data, C, set([]))
            
        # Estimate P(A, B | C)
        P_AB_given_C = empirical_estimate(self.data, A | B, C)

        # Estimate P(A | C)
        P_A_given_C = empirical_estimate(self.data, A, C)

        # Estimate P(B | C)
        P_B_given_C = empirical_estimate(self.data, B, C)

        assert np.isclose(1, sum(P_C.values()))
        assert np.isclose(2, sum(P_AB_given_C.values()))
        assert np.isclose(2, sum(P_A_given_C.values()))
        assert np.isclose(2, sum(P_B_given_C.values()))

        # Perform summation
        summation = 0
        for abc_val in itr.product(range(self.domain_size), repeat=len(A | B | C)):
            A_indices = sorted(list(A))
            B_indices = sorted(list(B))
            AB_indices = sorted(list(A | B))
            C_indices = sorted(list(C))
            a_val = tuple(np.array(abc_val)[A_indices])
            b_val = tuple(np.array(abc_val)[B_indices])
            ab_val = tuple(np.array(abc_val)[AB_indices])
            c_val = tuple(np.array(abc_val)[C_indices])
            assert len(a_val) == len(A)
            assert len(b_val) == len(B)
            assert len(ab_val) == len(A) + len(B)
            assert len(c_val) == len(C)

            P_C_key = (c_val, tuple())
            P_AB_given_C_key = (ab_val, c_val)
            P_A_given_C_key = (a_val, c_val)
            P_B_given_C_key = (b_val, c_val)
            summation += P_C[P_C_key]\
                * abs(P_AB_given_C[P_AB_given_C_key]
                      - P_A_given_C[P_A_given_C_key] * P_B_given_C[P_B_given_C_key])
        
        res = dict(
            statistic=summation,
            reject=summation >= self.epsilon,
        )
        return res
    
    def test_with_ctable(self, A: set, B: set, C: set):
        a_samples = vector2scalar(self.data[:, list(A)])
        b_samples = vector2scalar(self.data[:, list(B)])
        c_samples = vector2scalar(self.data[:, list(C)])
        ctab = crosstab(a_samples, b_samples, c_samples)
        n = self.data.shape[0]
        p_abc = ctab.count / n
        p_ac = p_abc.sum(axis=1)
        p_bc = p_abc.sum(axis=0)
        p_c = p_ac.sum(axis=0)

        term1 = p_abc
        term2 = p_ac[:, np.newaxis, :] * p_bc[np.newaxis, :, :] * p_c[np.newaxis, np.newaxis, :] ** -1
        distance = np.sum(np.abs(term1 - term2))

        res = dict(
            statistic=distance,
            reject=distance >= self.epsilon,
        )
        return res
    

if __name__ == "__main__":
    domain_size = 2
    dag = cd.DAG(arcs={(0, 1), (1, 2)})

    node_alphabets = {node: list(range(domain_size)) for node in dag.nodes}
    ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)

    samples = ddag.sample(10000)
    tester = PluginConditionalIndependenceTester(samples, 0.05, domain_size)

    A = {2}
    B = {0}
    C = {1}
    
    res = tester.test(A, B, C)
    print(res["statistic"])
    res = tester.test(B, A, C)
    print(res["statistic"])

    res_old = tester.old_test(A, B, C)
    print(res_old["statistic"])
    res_old = tester.old_test(B, A, C)
    print(res_old["statistic"])

    # ddag_est = cd.DiscreteDAG.fit(dag, samples, node_alphabets)
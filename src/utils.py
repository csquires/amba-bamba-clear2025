# === IMPORTS: BUILT-IN ===
import itertools as itr
from typing import List

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.stats.contingency import crosstab
from optimaladj.CausalGraph import CausalGraph

# === IMPORTS: LOCAL ===
from src.custom_types import Query


def vector2scalar(data: np.ndarray, alphabet_sizes: List[int]) -> np.ndarray:
    multiplier_vector = np.cumprod([1] + alphabet_sizes[:-1])
    scalarized_values = np.sum(data * multiplier_vector, axis=1)
    return scalarized_values

def estimate_alpha_lower_bound(
    data: np.ndarray,
    Sigma_X_sizes: List[int],
    Sigma_S_sizes: List[int],
    query: Query,
    S: set
):
    n = len(data)
    x_samples = vector2scalar(data[:, list([query.action])], Sigma_X_sizes)
    s_samples = vector2scalar(data[:, list(S)], Sigma_S_sizes)
    ctab = crosstab(x_samples, s_samples)
    p_xs = ctab.count / n
    p_x = p_xs.sum(axis=1)
    s_given_x = p_xs[query.action_value] / p_x[query.action_value]

    lb = float('inf')
    for val in s_given_x:
        lb = min(lb, val)
    return lb

def empirical_conditional(
    data: np.ndarray,
    query: Query,
    alphabet_sizes: List[int],
    alpha: float = 0
):
    dag = cd.DAG(arcs={(0, 1)})
    node_alphabets = {
        0: list(range(alphabet_sizes[query.action])),
        1: list(range(alphabet_sizes[query.target])),
    }
    xy_data = np.vstack((data[:, query.action], data[:, query.target])).T

    if alpha == 0:
        full_ddag = cd.DiscreteDAG.fit(dag, xy_data, node_alphabets=node_alphabets)
    else:
        full_ddag = cd.DiscreteDAG.fit(dag, xy_data, node_alphabets=node_alphabets, method="add_one_mle", alpha=alpha)

    return full_ddag.conditionals[1][query.target_value]


def adjust_by_S(
    data: np.ndarray,
    query: Query,
    S: set,
    alphabet_sizes: List[int],
    alpha: float = 0
):
    if len(S) == 0:
        res = empirical_conditional(data, query, alphabet_sizes, alpha=alpha)
        return res[query.target_value]
    
    n = data.shape[0]
    S_list = list(S)
    S_alphabet_sizes = [alphabet_sizes[i] for i in S_list]
    S_samples = vector2scalar(data[:, S_list], S_alphabet_sizes)

    # # specify levels
    # if alpha == 0:
    #     levels = None
    # else:
    #     levels_S = np.prod(S_alphabet_sizes)
    #     levels = [
    #         list(range(levels_S)), 
    #         list(range(alphabet_sizes[query.action])), 
    #         list(range(alphabet_sizes[query.target]))
    #     ]

    # compute base counts
    ctab = crosstab(S_samples, data[:, query.action], data[:, query.target], levels=None)
    counts_sxy = ctab.count
    counts_sx = np.sum(counts_sxy, axis=2)
    counts_s = np.sum(counts_sx, axis=1)

    # compute conditional probabilities, possibly with smoothing
    if alpha == 0:
        p_s = counts_s / n
        p_y_mid_sx_num = counts_sxy[:, query.action_value, query.target_value]
        p_y_mid_sx_den = counts_sx[:, query.action_value]
        p_y_mid_sx = p_y_mid_sx_num / p_y_mid_sx_den
        p_y_mid_sx[np.isnan(p_y_mid_sx)] = 1 / alphabet_sizes[query.target]

        unseen_term = 0
    else:
        S_all = np.prod(S_alphabet_sizes)
        
        p_s = (counts_s + alpha) / (n + alpha * S_all)
        p_y_mid_sx_num = counts_sxy[:, query.action_value, query.target_value] + alpha
        p_y_mid_sx_den = counts_sx[:, query.action_value] + alpha * alphabet_sizes[query.target]
        p_y_mid_sx = p_y_mid_sx_num / p_y_mid_sx_den

        S_unseen = S_all - len(set(S_samples))
        unseen_term = S_unseen * (alpha / (n + alpha * S_all)) * (1 / alphabet_sizes[query.target])

    terms = p_s * p_y_mid_sx
    est = sum(terms) + unseen_term
    return est


def adjust_by_S_old(
    data: np.ndarray,
    query: Query,
    S: set,
    alphabet_sizes: List[int],
    alpha: float = 0
) -> float:
    if len(S) == 0:
        return empirical_conditional(data, query, alphabet_sizes, alpha=alpha)

    S_list = list(S)
    S_alphabet_sizes = [alphabet_sizes[i] for i in S_list]
    dag = cd.DAG(nodes={0, 1, 2}, arcs={(0, 1), (0, 2), (1, 2)})

    # RE-LABEL VALUES OF S AND CREATE A NEW DATASET
    S_samples = vector2scalar(data[:, S_list], S_alphabet_sizes)
    small_data = np.vstack((S_samples, data[:, query.action], data[:, query.target])).T

    # FIT
    node_alphabets = {
        0: list(range(np.prod(S_alphabet_sizes))), 
        1: list(range(alphabet_sizes[query.action])), 
        2: list(range(alphabet_sizes[query.target]))
    }
    if alpha == 0:
        full_ddag = cd.DiscreteDAG.fit(dag, small_data, node_alphabets=node_alphabets)
    else:
        full_ddag = cd.DiscreteDAG.fit(dag, small_data, node_alphabets=node_alphabets, method="add_one_mle", alpha=alpha)


    # ESTIMATE INTERVENTIONAL QUERY
    iv_ddag = full_ddag.get_hard_interventional_dag(1, query.action_value)
    est_P_y_x = iv_ddag.get_marginal(2)[query.target_value]

    return est_P_y_x


def compute_causal_effects(
    dag: cd.DiscreteDAG, 
    query: Query
):
    g_iv = dag.get_hard_interventional_dag(query.action, query.action_value)
    P_y_x = g_iv.get_marginal(query.target)[query.target_value]

    return P_y_x


def compute_max_diff(
    true_causal_effects,
    estimated_causal_effects
):
    max_diff = float("-inf")
    for key in true_causal_effects:
        true_effect = true_causal_effects[key]
        est_effect = estimated_causal_effects[key]
        diff = abs(true_effect - est_effect)
        max_diff = max(max_diff, diff)
    return max_diff


if __name__ == "__main__":
    nnodes = 5
    dag = cd.rand.directed_erdos(nnodes, 0.05, random_order=False)
    dag.add_arc(nnodes-2, nnodes-1)
    dag.add_arcs_from({(nnodes-3, nnodes-2), (nnodes-3, nnodes-1)})
    
    node_alphabets = {i: list(range(i + 2)) for i in range(nnodes)}
    ddag = cd.rand.rand_discrete_dag(
        dag,
        node_alphabets
    )
    data = ddag.sample(100000)
    query = Query(nnodes-2, nnodes-1, 0, 1)
    
    alphabet_sizes = [i + 2 for i in range(nnodes)]
    estimated_causal_effect = adjust_by_S(data, query, dag.parents_of(nnodes-2), alphabet_sizes)
    true_causal_effects = compute_causal_effects(ddag, query)

    print(estimated_causal_effect)
    print(true_causal_effects)


def get_minimal_adjset(dag: cd.DAG, action, target):
    # Notes
    # -----
    # - We are currently using the optimaladj package
    #   - See https://arxiv.org/pdf/2201.02037
    #   - The package is based on networkx
    #   - Their implementation uses all_simple_paths, which is slow
    # - Alternatives:
    #   - We could re-implement van der Zander's linear-time algorithms
    #       - Papers:
    #           - http://proceedings.mlr.press/v115/van-der-zander20a/van-der-zander20a.pdf
    #           - https://arxiv.org/pdf/1803.00116
    #       - These are only in R
    #       - The only tricky part of the algorithm is that it requires a modification of the Bayes ball algorithm
    #       - I have the Bayes ball algorithm here, which we could take as starter code
    #           - https://github.com/uhlerlab/graphical_models/blob/30dcfd45d51bfbc482afe492ad26aad0c89b4cdc/graphical_models/classes/dags/dag.py#L1861

    G = CausalGraph()
    G.add_nodes_from(dag.nodes)
    G.add_edges_from(dag.arcs)
    L = []  # the conditioning set if doing CATE estimation, empty for us
    N = list(dag.nodes)  # observed nodes

    assert len(dag.nodes) == len(G.nodes)
    assert action in G.nodes
    assert target in G.nodes

    # optimal_adj_set, optimal_minimal_adj_set, optimal_minimum_adj_set
    Smin = G.optimal_adj_set(action, target, L, N)
    # Smin = G.optimal_minimal_adj_set(action, target, L, N)
    # Smin = G.optimal_minimum_adj_set(action, target, L, N)
    return Smin

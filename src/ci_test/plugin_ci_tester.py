# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.stats.contingency import crosstab
from operator import mul
from einops import rearrange
from functools import reduce

# === IMPORTS: LOCAL ===
from src.ci_test.custom_ci_tester import ConditionalIndependenceTester
from src.utils import vector2scalar


class PluginConditionalIndependenceTester(ConditionalIndependenceTester):
    def __init__(self, data, epsilon, alphabet_sizes, univariate=False):
        super().__init__(data, epsilon, alphabet_sizes)
        self.univariate = univariate

    def test(self, A: set, B: set, C: set):
        if self.univariate:
            return self.test_univariate(A, B, C)
        else:
            return self.test_multivariate(A, B, C)

    def test_multivariate(self, A: set, B: set, C: set):
        A_alphabet_sizes = [self.alphabet_sizes[i] for i in A]
        B_alphabet_sizes = [self.alphabet_sizes[i] for i in B]
        C_alphabet_sizes = [self.alphabet_sizes[i] for i in C]
        a_samples = vector2scalar(self.data[:, list(A)], A_alphabet_sizes)
        b_samples = vector2scalar(self.data[:, list(B)], B_alphabet_sizes)
        c_samples = vector2scalar(self.data[:, list(C)], C_alphabet_sizes)
        ctab = crosstab(a_samples, b_samples, c_samples)
        n = self.data.shape[0]
        p_abc = ctab.count / n
        p_ac = p_abc.sum(axis=1)
        p_bc = p_abc.sum(axis=0)
        p_c = p_ac.sum(axis=0)

        term1 = p_abc
        term2 = np.einsum("ac,bc,c->abc", p_ac, p_bc, p_c ** -1)
        distance = np.sum(np.abs(term1 - term2))

        res = dict(
            statistic=distance,
            reject=distance >= self.epsilon,
        )
        return res
    
    def test_univariate(self, A: set, B: set, C: set):
        statistic = 0
        for b in B:
            res = self.test_multivariate(A, {b}, C)
            statistic += res["statistic"] / len(B)
        return dict(
            statistic=statistic,
            reject=statistic >= self.epsilon
        )
    

class MemoizedPluginConditionalIndependenceTester(ConditionalIndependenceTester):
    def __init__(self, data: np.ndarray, epsilon: float, domain_size: int, univariate=False):
        self.data = data
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.univariate = univariate

        # Compute probabilities
        nsamples = self.data.shape[0]
        self.nvariables = self.data.shape[1]
        self.marginals = dict()
        self.full_marginal = crosstab(*[self.data[:, i] for i in range(self.nvariables)]).count / nsamples

    def compute_marginal_old(self, canonical_order):
        canonical_marginal = crosstab(*(self.data[:, i] for i in canonical_order)).count / self.data.shape[0]
        return canonical_marginal

    def compute_marginal_new(self, canonical_order):
        pattern = "".join(chr(97+i) for i in range(self.nvariables))
        pattern += "->"
        pattern += "".join(chr(97+i) for i in canonical_order)
        canonical_marginal = np.einsum(pattern, self.full_marginal)
        return canonical_marginal

    def get_marginal(self, subsets: tuple[tuple[int]]):
        if len(subsets) != 3:
            raise ValueError("This method only supports 3 subsets")
        # === example ===
        ### INPUTS
        # alphabet sizes: {0: a0, 1: a1, ...}
        # subsets = ((6, 8), (2, 3, 4), (0, 1))
        ### ORDERS
        # original_order = [6, 8, 2, 3, 4, 0, 1]
        # canonical_order = (0, 1, 2, 3, 4, 6, 8)
        ### RESHAPING
        # canonical_marginal shape: a0 x a1 x a2 x a3 x a4 x a6 x a8
        # new_order = [5, 6, 2, 3, 4, 0, 1]
        # new_shape = [(a6*a8), (a2*a3*a4), (a0*a1)]

        # PUT INDICES INTO A CANONICAL ORDER
        original_order = [i for subset in subsets for i in subset]
        canonical_order = tuple(sorted(original_order))

        # RETRIEVE MARGINAL IF ALREADY COMPUTED
        if canonical_order in self.marginals:
            canonical_marginal = self.marginals[canonical_order]
        # OTHERWISE, COMPUTE MARGINAL AND STORE IT
        else:
            canonical_marginal = self.compute_marginal_new(canonical_order)
            self.marginals[canonical_order] = canonical_marginal

        # FORMAT INTO THE EXPECTED SHAPE
        # fix the ordering
        new_order = [canonical_order.index(i) for i in original_order]
        marginal_ordered = np.transpose(canonical_marginal, new_order)
        # fix the grouping
        A, B, C = subsets[0], subsets[1], subsets[2]
        shape_A = reduce(mul, marginal_ordered.shape[:len(A)], 1)
        shape_B = reduce(mul, marginal_ordered.shape[len(A):len(A) + len(B)], 1)
        shape_C = reduce(mul, marginal_ordered.shape[len(A) + len(B): len(A) + len(B) + len(C)], 1)
        marginal = marginal_ordered.reshape((shape_A, shape_B, shape_C), order="F")  # I forget why/if "F" is necessary

        return marginal
    
    def test(self, A: set, B: set, C: set):
        if self.univariate:
            return self.test_univariate(A, B, C)
        else:
            return self.test_multivariate(A, B, C)

    def test_multivariate(self, A: set, B: set, C: set):  # (not assuming faithfulness)
        A, B, C = tuple(sorted(A)), tuple(sorted(B)), tuple(sorted(C))
        # note: this is quite simple because of the reshaping done in get_marginal
        #   it might be faster to remove the reshaping there and make the code here more complex
        #   Example
        #   A = {b}, B = {a, d, e}, C = {c, f}
        #   p_abc has shape (a, b, c, d, e, f)
        #   p_ac = np.einsum("abcde->bcf", p_abc)
        #   p_bc = np.einsum("abcde->adecf", p_abc)
        #   p_c = np.einsum("bcf->cf", p_ac)
        #   from there, not sure how to do nonzero indexing in the best way
        p_abc = self.get_marginal((A, B, C))
        p_ac = p_abc.sum(axis=1)
        p_bc = p_abc.sum(axis=0)
        p_c = p_ac.sum(axis=0)
        nonzero_ixs = np.where(p_c != 0)[0]

        term1 = p_abc[:, :, nonzero_ixs]
        term2 = np.einsum("ac,bc,c->abc", p_ac[:, nonzero_ixs], p_bc[:, nonzero_ixs], p_c[nonzero_ixs] ** -1)
        distance = np.sum(np.abs(term1 - term2))

        res = dict(
            statistic=distance,
            reject=distance >= self.epsilon,
        )
        return res
    
    def test_univariate(self, A: set, B: set, C: set):  # (assuming faithfulness) 
        statistic = 0
        for b in B:
            res = self.test_multivariate(A, {b}, C)
            statistic += res["statistic"] / len(B)
        return dict(
            statistic=statistic,
            reject=statistic >= self.epsilon
        )
        
    

if __name__ == "__main__":

    # === FIRST TEST: CONDITIONALLY INDEPENDENT ===
    domain_size = 2
    dag = cd.DAG(arcs={(0, 1), (1, 2)})
    node_alphabets = {node: list(range(domain_size)) for node in dag.nodes}
    ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)

    samples = ddag.sample(10000)
    tester = PluginConditionalIndependenceTester(samples, 0.05, alphabet_sizes=[domain_size]*3)

    A = {2}
    B = {0}
    C = {1}
    
    res1_ab = tester.test(A, B, C)
    res1_ba = tester.test(B, A, C)
    print(res1_ab["statistic"], res1_ba["statistic"])


    # === SECOND TEST: MEMOIZED CONDITIONAL INDEPENDENCE TESTER ===
    dag = cd.DAG(nodes=set(range(10)))
    node_alphabets = {node: list(range(node+2)) for node in dag.nodes}
    # node_alphabets = {node: list(range(2)) for node in dag.nodes}
    ddag = cd.rand.rand_discrete_dag(dag, node_alphabets)
    samples = ddag.sample(10000)
    alphabet_sizes = [2 for _ in range(10)]
    tester = PluginConditionalIndependenceTester(samples, 0.05, alphabet_sizes)
    memo_tester = MemoizedPluginConditionalIndependenceTester(samples, 0.05, alphabet_sizes)
    # sanity check: all different sizes and not using all variables
    A = {6, 8}
    B = {2, 4, 3}
    C = {1, 0}

    res2_ab = tester.test(A, B, C)
    res2_ba = tester.test(B, A, C)
    print(res2_ab["statistic"], res2_ba["statistic"])

    memo_res2_ab = memo_tester.test(A, B, C)
    memo_res2_ba = memo_tester.test(B, A, C)
    print(memo_res2_ab["statistic"], memo_res2_ab["statistic"])

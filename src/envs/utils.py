import random
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np
from numba import jit


class EdgeType(Enum):
    UNIFORM = 1
    DISCRETE = 2
    RANDOM = 3


class RewardSignal(Enum):
    DENSE = 1
    BLS = 2
    SINGLE = 3
    CUSTOM_BLS = 4


class ExtraAction(Enum):
    PASS = 1
    RANDOMISE = 2
    NONE = 3


class OptimisationTarget(Enum):
    DSP = 1
    ENERGY = 2
    MTDS = 3  # 最小总支配集问题


class SpinBasis(Enum):
    SIGNED = 1
    BINARY = 2


class Observable(Enum):
    # Local observations that differ between nodes.
    SPIN_STATE = 1
    IMMEDIATE_REWARD_AVAILABLE = 2
    TIME_SINCE_FLIP = 3
    NEIGHBOR_COVERAGE = 4

    # Global observations that are the same for all nodes.
    EPISODE_TIME = 5
    TERMINATION_IMMANENCY = 6
    NUMBER_OF_GREEDY_ACTIONS_AVAILABLE = 7
    DISTANCE_FROM_BEST_SCORE = 8
    DISTANCE_FROM_BEST_STATE = 9
    TDS_VALIDITY_RATIO = 10
    DLP_BOUND = 11


# 更新默认可观测集合
DEFAULT_OBSERVABLES = [
    Observable.SPIN_STATE,
    Observable.IMMEDIATE_REWARD_AVAILABLE,
    Observable.TIME_SINCE_FLIP,
    Observable.DISTANCE_FROM_BEST_SCORE,
    Observable.DISTANCE_FROM_BEST_STATE,
    Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE,
    Observable.TERMINATION_IMMANENCY,
    Observable.NEIGHBOR_COVERAGE,
    Observable.TDS_VALIDITY_RATIO,
    Observable.DLP_BOUND
]


# ==================== MTDS验证和计算函数 ====================

def apply_mtds_remedy(adj_matrix, spins, spin_basis=SpinBasis.SIGNED):
    n = adj_matrix.shape[0]
    remedied_spins = spins.copy()

    # 1. 统一转换为集合 S (包含在 TDS 中的节点索引)
    if spin_basis == SpinBasis.BINARY:
        # 0 是选中 (In Set)
        S = set(np.where(remedied_spins == 0)[0])
    else:  # SIGNED
        # -1 是选中 (In Set)
        S = set(np.where(remedied_spins == -1)[0])

    original_size = len(S)

    # --- Phase 1: Domination Property (所有节点必须被 S 中的节点支配) ---
    # coverage[v] 表示 v 有多少个邻居在 S 中
    # 使用矩阵乘法加速: coverage = A @ indicator_vector
    indicator = np.zeros(n)
    indicator[list(S)] = 1
    coverage = adj_matrix @ indicator

    # 找出未被支配的点 (coverage == 0)
    undominated = np.where(coverage == 0)[0]

    for v in undominated:
        # 重新检查覆盖 (因为之前的循环可能已经覆盖了它)
        neighbors = np.where(adj_matrix[v, :] != 0)[0]
        neighbors = neighbors[neighbors != v]

        if len(set(neighbors) & S) > 0:
            continue  # 已经被新加入的点覆盖了

        # 贪心选择：选度数最大的邻居
        if len(neighbors) > 0:
            degrees = np.sum(adj_matrix[neighbors, :] != 0, axis=1)
            best_neighbor = neighbors[np.argmax(degrees)]
            S.add(best_neighbor)

    # --- Phase 2: Total Property (S 中的每个节点必须被 S 中的其他节点支配) ---
    S_list = list(S)
    for u in S_list:
        neighbors = np.where(adj_matrix[u, :] != 0)[0]
        neighbors = neighbors[neighbors != u]

        # 检查 u 是否被 S 中的其他点支配
        if len(set(neighbors) & S) == 0:
            # u 是 S 中的孤立点，必须拉一个邻居进来
            if len(neighbors) > 0:
                degrees = np.sum(adj_matrix[neighbors, :] != 0, axis=1)
                best_neighbor = neighbors[np.argmax(degrees)]
                S.add(best_neighbor)

    # 转换回 spins 格式
    is_modified = (len(S) != original_size)

    if spin_basis == SpinBasis.BINARY:
        new_spins = np.ones(n)  # 1: 不在 S
        new_spins[list(S)] = 0  # 0: 在 S
    else:
        new_spins = np.ones(n)  # 1: 不在 S
        new_spins[list(S)] = -1  # -1: 在 S

    return new_spins, is_modified


def calculate_dual_lp_bound(adj_matrix):
    """
    计算 MTDS 的简单 LP 对偶下界。
    """
    n = adj_matrix.shape[0]
    y = np.zeros(n)
    nodes = np.arange(n)
    np.random.shuffle(nodes)

    constraints = np.ones(n)  # 每个节点的容量剩余

    for v in nodes:
        neighbors = np.where(adj_matrix[v, :] != 0)[0]
        neighbors = neighbors[neighbors != v]

        if len(neighbors) == 0: continue

        min_slack = np.min(constraints[neighbors])

        if min_slack > 0:
            val = min_slack
            y[v] += val
            constraints[neighbors] -= val

    return np.sum(y)


def is_total_dominating_set(adj_matrix, spins, spin_basis=SpinBasis.SIGNED):
    """
    检查给定的自旋配置是否是总支配集 (向量化加速版)
    """
    n = adj_matrix.shape[0]

    # 1. 构造指示向量 x (1 if in S, 0 otherwise)
    if spin_basis == SpinBasis.BINARY:
        x = (spins == 0).astype(float)
    else:
        x = (spins == -1).astype(float)

    S_indices = np.where(x == 1)[0]

    if len(S_indices) == 0: return False

    # 2. 计算每个节点被 S 覆盖的次数: c = A @ x
    # 假设 adj_matrix 对角线为 0 (无自环)
    coverage = adj_matrix @ x

    # 条件1：支配性 (Domination)
    # 所有节点的 coverage 必须 >= 1
    # 注意：这里需要排除孤立点的情况，如果原图有孤立点，它永远无法被支配。
    # 我们假设图本身是无孤立点的。
    if np.any(coverage == 0):
        # 检查是否是因为图本身有孤立点(度为0)
        degrees = np.sum(adj_matrix != 0, axis=1)
        # 如果存在 coverage==0 且 degree > 0 的点，则非法
        if np.any((coverage == 0) & (degrees > 0)):
            return False
        # 如果 coverage==0 的点都是孤立点，且被选中了...
        # 但孤立点不可能被别人支配。根据定义，孤立点无法构成 TDS。
        # 所以只要有 coverage==0 就是 False (对于 NI 图)
        return False

    # 条件2：全总性 (Total Property)
    # S 中的每个节点，必须被 S 中的其他节点支配
    # 即对于所有 u in S, coverage[u] >= 1
    # 由于 coverage 已经是 A @ x，它计算的就是来自“邻居”的贡献（不含自己，只要A对角线为0）
    # 所以直接检查 S 中的点的 coverage 即可

    coverage_in_S = coverage[S_indices]
    if np.any(coverage_in_S == 0):
        return False

    return True


def calculate_greedy_dual_bound(adj_matrix):
    n = adj_matrix.shape[0]
    y = np.zeros(n)
    constraint_sums = np.zeros(n)
    indices = np.arange(n)
    np.random.shuffle(indices)

    dual_obj = 0.0

    for v in indices:
        neighbors = np.where(adj_matrix[v, :] != 0)[0]
        neighbors = neighbors[neighbors != v]

        max_increase = 1.0
        for u in neighbors:
            slack = 1.0 - constraint_sums[u]
            if slack < max_increase:
                max_increase = slack

        if max_increase > 0:
            y[v] += max_increase
            dual_obj += max_increase
            for u in neighbors:
                constraint_sums[u] += max_increase

    return dual_obj


def calculate_mtds_size(adj_matrix, spins, spin_basis=SpinBasis.SIGNED):
    if not is_total_dominating_set(adj_matrix, spins, spin_basis):
        return -1

    if spin_basis == SpinBasis.BINARY:
        return int(np.sum(spins == 0))
    else:
        return int(np.sum(spins == -1))


def get_neighbor_coverage(adj_matrix, spins, spin_basis=SpinBasis.SIGNED):
    """
    [高性能版] 计算每个顶点的邻接覆盖度
    利用矩阵乘法 A @ x 替代循环，速度提升显著
    """
    # 构造指示向量 x: 选中为 1，未选中为 0
    if spin_basis == SpinBasis.BINARY:
        x = (spins == 0).astype(float)
    else:
        x = (spins == -1).astype(float)

    # 矩阵乘法计算覆盖数
    # coverage[i] = sum(A[i][j] * x[j]) = i 的邻居中被选中的个数
    # 前提：adj_matrix 对角线为 0 (GraphGenerator 已保证)
    coverage = adj_matrix @ x

    return coverage


def get_tds_validity_ratio(adj_matrix, spins, spin_basis=SpinBasis.SIGNED):
    """
    [高性能版] 计算 TDS 有效性比率
    """
    n = adj_matrix.shape[0]
    if n == 0: return 0.0

    # 1. 构造指示向量
    if spin_basis == SpinBasis.BINARY:
        x = (spins == 0).astype(float)
    else:
        x = (spins == -1).astype(float)

    # 2. 计算覆盖
    coverage = adj_matrix @ x

    # 3. 统计有效节点 (coverage >= 1)
    # 排除图本身的孤立点(度为0)，避免它们拉低分数
    degrees = np.sum(adj_matrix != 0, axis=1)
    non_isolated_mask = (degrees > 0)

    valid_mask = (coverage >= 1) & non_isolated_mask
    valid_count = np.sum(valid_mask)

    # 分母只考虑非孤立点（或者全图，取决于定义，这里用全图简单归一化）
    # 为了梯度平滑，直接除以 n
    return valid_count / n


# ==========================================
# MTDS 剪枝工具函数
# ==========================================
def prune_solution_fast(adj_list, solution_set):
    """
    严格的 MTDS 剪枝算法（保证 Total Dominating）
    """
    if not solution_set:
        return set()

    n = len(adj_list)
    S = set(solution_set)

    # dom_count[v] = v 被 S 中多少邻居支配
    dom_count = [0] * n

    for u in S:
        for v in adj_list[u]:
            dom_count[v] += 1

    # 启发式：先删“度小的 / 冗余的”
    candidates = sorted(S, key=lambda x: len(adj_list[x]))

    for u in candidates:
        if u not in S:
            continue

        # 1️⃣ u 自己还能被支配吗？
        still_dominated = False
        for w in adj_list[u]:
            if w in S and w != u:
                still_dominated = True
                break
        if not still_dominated:
            continue

        # 2️⃣ u 的所有邻居还能被支配吗？
        can_remove = True
        for v in adj_list[u]:
            if dom_count[v] <= 1:
                can_remove = False
                break

        if not can_remove:
            continue

        # ✅ 安全删除 u
        S.remove(u)
        for v in adj_list[u]:
            dom_count[v] -= 1
    return S


class GraphGenerator(ABC):

    def __init__(self, n_spins, edge_type, biased=False):
        self.n_spins = n_spins
        self.edge_type = edge_type
        self.biased = biased

    def pad_matrix(self, matrix):
        dim = matrix.shape[0]
        m = np.zeros((dim + 1, dim + 1))
        m[:-1, :-1] = matrix
        return m

    def pad_bias(self, bias):
        return np.concatenate((bias, [0]))

    @abstractmethod
    def get(self, with_padding=False):
        raise NotImplementedError


class RandomNonIsolatedGraphGenerator(GraphGenerator):
    """
    随机生成无孤立点的图(每个节点度数至少为1)的生成器。
    """

    def __init__(self, n_spins=20, edge_type=EdgeType.DISCRETE, biased=False,
                 max_attempts=10, density_range=(0.2, 1.0)):
        super().__init__(n_spins, edge_type, biased)
        self.max_attempts = max_attempts
        self.density_range = density_range

        if self.edge_type == EdgeType.UNIFORM:
            self.get_w = lambda: 1
        elif self.edge_type == EdgeType.DISCRETE:
            self.get_w = lambda: np.random.choice([1, 0])
        elif self.edge_type == EdgeType.RANDOM:
            self.get_w = lambda: np.random.uniform(-1, 1)
        else:
            raise NotImplementedError(f"EdgeType {self.edge_type} not supported")

        # 修复权重函数
        if self.edge_type == EdgeType.DISCRETE:
            self.get_fix_w = lambda: 1
        elif self.edge_type == EdgeType.RANDOM:
            self.get_fix_w = lambda: np.random.choice([-1, 1]) * np.random.uniform(0.1, 1.0)
        else:
            self.get_fix_w = self.get_w

    def get(self, with_padding=False):
        matrix = None
        # 随机选择当前图的密度
        density = np.random.uniform(*self.density_range)

        for attempt in range(self.max_attempts):
            # Numpy 向量化生成
            mask = np.random.rand(self.n_spins, self.n_spins) < density
            mask = np.triu(mask, k=1)

            if self.edge_type == EdgeType.UNIFORM:
                weights = np.ones((self.n_spins, self.n_spins))
            elif self.edge_type == EdgeType.DISCRETE:
                weights = np.random.choice([0, 1], size=(self.n_spins, self.n_spins))
            elif self.edge_type == EdgeType.RANDOM:
                weights = np.random.uniform(-1, 1, size=(self.n_spins, self.n_spins))

            upper = np.multiply(mask, weights)
            matrix = upper + upper.T

            degrees = np.sum(matrix != 0, axis=0)
            isolated_nodes = np.where(degrees == 0)[0]

            # 修复孤立点
            for node in isolated_nodes:
                potential_neighbors = [i for i in range(self.n_spins) if i != node]
                if not potential_neighbors: continue  # n=1 case
                neighbor = np.random.choice(potential_neighbors)
                w = self.get_fix_w()
                matrix[node, neighbor] = w
                matrix[neighbor, node] = w

            degrees = np.sum(matrix != 0, axis=0)
            if np.all(degrees >= 1):
                break
        else:
            print(f"Warning: Failed to generate non-isolated graph. Using fully connected.")
            matrix = np.ones((self.n_spins, self.n_spins))
            np.fill_diagonal(matrix, 0)

        matrix = self.pad_matrix(matrix) if with_padding else matrix

        if self.biased:
            bias = np.array([
                self.get_w() if np.random.uniform() < density else 0
                for _ in range(self.n_spins)
            ])
            bias = self.pad_bias(bias) if with_padding else bias
            return matrix, bias
        else:
            return matrix


class RandomGridGraphGenerator(GraphGenerator):
    """
    随机网格图生成器。
    自动根据 n_spins 计算最接近正方形的 (h, w) 维度。
    """

    def __init__(self, n_spins=20, edge_type=EdgeType.UNIFORM, biased=False, periodic=False):
        super().__init__(n_spins, edge_type, biased)
        self.periodic = periodic

        # 定义权重生成函数
        if self.edge_type == EdgeType.UNIFORM:
            self.get_w = lambda: 1
        elif self.edge_type == EdgeType.DISCRETE:
            # 离散权重通常用于自旋玻璃问题 (+1/-1)
            self.get_w = lambda: np.random.choice([1, -1])
        elif self.edge_type == EdgeType.RANDOM:
            self.get_w = lambda: np.random.uniform(-1, 1)
        else:
            raise NotImplementedError(f"EdgeType {self.edge_type} not supported for Grid")

    def _get_dims(self, n):
        """寻找 n 最接近正方形的因子对 (h, w)"""
        sqrt_n = int(np.sqrt(n))
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n  # 无法分解时退化为链状图

    def get(self, with_padding=False):
        # 1. 计算维度
        h, w = self._get_dims(self.n_spins)

        # 2. 生成 Grid 拓扑
        g = nx.grid_2d_graph(h, w, periodic=self.periodic)
        adj = nx.to_numpy_array(g)

        # 3. 如果需要非均匀权重，则应用权重
        if self.edge_type != EdgeType.UNIFORM:
            # 获取上三角部分的边索引
            rows, cols = np.nonzero(np.triu(adj))
            for r, c in zip(rows, cols):
                w_val = self.get_w()
                adj[r, c] = w_val
                adj[c, r] = w_val

        # 确保对角线为0
        np.fill_diagonal(adj, 0)

        # 4. 处理 Padding 和 Bias
        matrix = self.pad_matrix(adj) if with_padding else adj

        if self.biased:
            # Grid 这里的 bias 逻辑简单处理，随机生成
            density = 0.5
            bias = np.array([
                self.get_w() if np.random.uniform() < density else 0
                for _ in range(self.n_spins)
            ])
            bias = self.pad_bias(bias) if with_padding else bias
            return matrix, bias
        else:
            return matrix


class RandomGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if self.edge_type == EdgeType.UNIFORM:
            self.get_w = lambda: 1
        elif self.edge_type == EdgeType.DISCRETE:
            self.get_w = lambda: np.random.choice([1, 0])
        elif self.edge_type == EdgeType.RANDOM:
            self.get_w = lambda: np.random.uniform(-1, 1)
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        g_size = self.n_spins
        density = np.random.uniform()
        matrix = np.zeros((g_size, g_size))
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    w = self.get_w()
                    matrix[i, j] = w
                    matrix[j, i] = w
        matrix = self.pad_matrix(matrix) if with_padding else matrix

        if self.biased:
            bias = np.array([self.get_w() if np.random.uniform() < density else 0 for _ in range(self.n_spins)])
            bias = self.pad_bias(bias) if with_padding else bias
            return matrix, bias
        else:
            return matrix


class RandomErdosRenyiGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, p_connection=[0.1, 0], edge_type=EdgeType.DISCRETE):
        super().__init__(n_spins, edge_type, False)

        if type(p_connection) not in [list, tuple]:
            p_connection = [p_connection, 0]
        assert len(p_connection) == 2, "p_connection must have length 2"
        self.p_connection = p_connection

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):

        p = np.clip(np.random.normal(*self.p_connection), 0, 1)

        g = nx.erdos_renyi_graph(self.n_spins, p)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj


class RandomBarabasiAlbertGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, m_insertion_edges=4, edge_type=EdgeType.DISCRETE):
        super().__init__(n_spins, edge_type, False)

        self.m_insertion_edges = m_insertion_edges

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):

        g = nx.barabasi_albert_graph(self.n_spins, self.m_insertion_edges)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj


class RandomRegularGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, d_node=[2, 0], edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if type(d_node) not in [list, tuple]:
            d_node = [d_node, 0]
        assert len(d_node) == 2, "k_neighbours must have length 2"
        self.d_node = d_node

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        k = np.clip(int(np.random.normal(*self.d_node)), 0, self.n_spins)

        g = nx.random_regular_graph(k, self.n_spins)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        if not self.biased:
            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj


class SingleGraphGenerator(GraphGenerator):

    def __init__(self, matrix, bias=None):

        n_spins = matrix.shape[0]

        if np.isin(matrix, [0, 1]).all():
            edge_type = EdgeType.UNIFORM
        elif np.isin(matrix, [0, -1, 1]).all():
            edge_type = EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, bias is not None)

        self.matrix = matrix
        self.bias = bias

    def get(self, with_padding=False):

        m = self.pad_matrix(self.matrix) if with_padding else self.matrix

        if self.biased:
            b = self.pad_bias(self.bias) if with_padding else self.bias
            return m, b
        else:
            return m


class SetGraphGenerator(GraphGenerator):

    def __init__(self, matrices, biases=None, ordered=False):

        if len(set([m.shape[0] - 1 for m in matrices])) == 1:
            n_spins = matrices[0].shape[0]
        else:
            raise NotImplementedError("All graphs in SetGraphGenerator must have the same dimension.")

        if all([np.isin(m, [0, 1]).all() for m in matrices]):
            edge_type = EdgeType.UNIFORM
        elif all([np.isin(m, [0, -1, 1]).all() for m in matrices]):
            edge_type = EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, biases is not None)

        if not self.biased:
            self.graphs = matrices
        else:
            assert len(matrices) == len(biases), "Must pass through the same number of matrices and biases."
            assert all([len(b) == self.n_spins + 1 for b in
                        biases]), "All biases and must have the same dimension as the matrices."
            self.graphs = list(zip(matrices, biases))

        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self, with_padding=False):
        if self.ordered:
            m = self.graphs[self.i]
            self.i = (self.i + 1) % len(self.graphs)
        else:
            m = random.sample(self.graphs, k=1)[0]
        return self.pad_matrix(m) if with_padding else m


class HistoryBuffer():
    def __init__(self):
        self.buffer = {}
        self.current_action_hist = set([])
        self.current_action_hist_len = 0

    def update(self, action):
        new_action_hist = self.current_action_hist.copy()
        if action in self.current_action_hist:
            new_action_hist.remove(action)
            self.current_action_hist_len -= 1
        else:
            new_action_hist.add(action)
            self.current_action_hist_len += 1
        try:
            list_of_states = self.buffer[self.current_action_hist_len]
            if new_action_hist in list_of_states:
                self.current_action_hist = new_action_hist
                return False
        except KeyError:
            list_of_states = []

        list_of_states.append(new_action_hist)
        self.current_action_hist = new_action_hist
        self.buffer[self.current_action_hist_len] = list_of_states
        return True
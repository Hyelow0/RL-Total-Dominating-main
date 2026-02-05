import random
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy, copy
from operator import matmul

import numpy as np
import torch.multiprocessing as mp
from numba import jit, float64, int64, int32

from src.envs.utils import (EdgeType,
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            SpinBasis,
                            DEFAULT_OBSERVABLES,
                            GraphGenerator,
                            RandomGraphGenerator,
                            HistoryBuffer,
                            is_total_dominating_set,
                            calculate_mtds_size,
                            get_neighbor_coverage,
                            get_tds_validity_ratio,
                            calculate_dual_lp_bound,
                            calculate_greedy_dual_bound)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class SpinSystemFactory(object):

    @staticmethod
    def get(graph_generator=None,
            max_steps=20,
            observables=DEFAULT_OBSERVABLES,
            reward_signal=RewardSignal.DENSE,
            extra_action=ExtraAction.PASS,
            optimisation_target=OptimisationTarget.ENERGY,
            spin_basis=SpinBasis.SIGNED,
            norm_rewards=True,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None,  # None means an infinite horizon.
            stag_punishment=None,  # None means no punishment for re-visiting states.
            basin_reward=None,  # None means no reward for reaching a local minima.
            reversible_spins=True,  # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            init_snap=None,
            seed=None,
            ifweight=False,
            mtds_constraint_penalty=3.0,
            size_penalty=0.1):

        if graph_generator.biased:
            return SpinSystemBiased(graph_generator, max_steps,
                                    observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                    norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                    reversible_spins,
                                    init_snap, seed, ifweight, mtds_constraint_penalty, size_penalty)
        else:
            return SpinSystemUnbiased(graph_generator, max_steps,
                                      observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                      norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                      reversible_spins,
                                      init_snap, seed, ifweight, mtds_constraint_penalty, size_penalty)


class SpinSystemBase(ABC):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''
    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            return np.random.choice(self.actions, n)

    class observation_space():
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal=RewardSignal.DENSE,
                 extra_action=ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.MTDS,
                 spin_basis=SpinBasis.SIGNED,
                 norm_rewards=True,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_spins=True,
                 init_snap=None,
                 seed=None,
                 ifweight=False,
                 mtds_constraint_penalty=3.0,
                 size_penalty=1.0,
                 immediate_reward_mode="RL"):

        '''
        Init method.
        '''
        self.mtds_constraint_penalty = mtds_constraint_penalty
        self.size_penalty = size_penalty
        self.optimisation_target = optimisation_target
        self.stage_two_mode = False

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."

        self.observables = list(enumerate(observables))
        self.extra_action = extra_action

        if graph_generator != None:
            assert isinstance(graph_generator,
                              GraphGenerator), "graph_generator must be a GraphGenerator implementation."
            self.gg = graph_generator
        else:
            # provide a default graph generator if one is not passed
            self.gg = RandomGraphGenerator(n_spins=20,
                                           edge_type=EdgeType.DISCRETE,
                                           biased=False,
                                           extra_action=(extra_action != extra_action.NONE))

        self.n_spins = self.gg.n_spins  # Total number of spins in episode
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_spins
        if extra_action != ExtraAction.NONE:
            self.n_actions += 1

        self.action_space = self.action_space(self.n_actions)
        self.observation_space = self.observation_space(self.n_spins, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.spin_basis = spin_basis
        self.immediate_reward_mode = immediate_reward_mode

        # MTDS feasibility check
        if self.optimisation_target == OptimisationTarget.MTDS:
            self._validate_mtds_feasibility()

        self.memory_length = memory_length
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_spins = reversible_spins

        self.reset()

        self.score = self.calculate_score()
        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        # Initialization Logic Corrected
        if self.optimisation_target == OptimisationTarget.MTDS:
            self.best_tds_size = np.inf
            self.best_score = np.inf
            self.best_obs_score = np.inf
        else:
            self.best_score = self.score
            self.best_obs_score = self.score

        self.best_spins = self.state[0, : self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        if init_snap != None:
            self.load_snapshot(init_snap)

        self.random_weight = np.array(np.random.randint(1, 10, size=self.n_spins))
        self.ifweight = ifweight

    def _validate_mtds_feasibility(self):
        degrees = np.sum(self.matrix != 0, axis=1)
        isolated = np.where(degrees == 0)[0]
        if len(isolated) > 0:
            print(f"Warning: Isolated nodes detected in graph: {isolated.tolist()}")

    def reset(self, spins=None, reset_weight=True):
        """
        Explanation here
        """
        if reset_weight:
            self.random_weight = np.array(np.random.randint(1, 10, size=self.n_spins))
        self.current_step = 0
        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()

        # validate graph in the mtds mode
        if self.optimisation_target == OptimisationTarget.MTDS:
            try:
                self._validate_mtds_feasibility()
            except ValueError:
                return self.reset(spins, reset_weight)

        self._reset_graph_observables()
        # Calculate max local reward for normalization
        spinsOne = np.array([1] * self.n_spins)
        local_rewards_available = self.get_immeditate_rewards_avaialable(spinsOne)
        # Avoid empty array issues
        if np.any(local_rewards_available):
            self.max_local_reward_available = np.max(np.abs(local_rewards_available)) + 1e-6
        else:
            self.max_local_reward_available = 1.0

        self.state = self._reset_state(spins)
        # self.score = 0
        # 初始 Score 计算：MTDS 使用 Raw Size
        if self.optimisation_target == OptimisationTarget.MTDS:
            if self.spin_basis == SpinBasis.BINARY:
                self.score = np.sum(self.state[0, :self.n_spins] == 0)
            else:
                self.score = np.sum(self.state[0, :self.n_spins] == -1)
        else:
            self.score = self.calculate_score()

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        # Reset Best Scores
        if self.optimisation_target == OptimisationTarget.MTDS:
            self.best_tds_size = np.inf
            self.best_score = np.inf
            self.best_obs_score = np.inf

            # 如果初始状态恰好合法，更新 Best
            validity = get_tds_validity_ratio(self.matrix, self.state[0, :self.n_spins], self.spin_basis)
            if validity >= 1.0:
                self.best_score = self.score
                self.best_obs_score = self.score
        else:
            self.best_score = self.score
            self.best_obs_score = self.score

        self.best_spins = self.state[0, :self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        if self.memory_length is not None:
            # Initialize with appropriate worst-case values
            init_val = np.inf if self.optimisation_target == OptimisationTarget.MTDS else -np.inf
            self.score_memory = np.array([init_val] * self.memory_length)
            self.spins_memory = np.array([self.best_spins] * self.memory_length)
            self.idx_memory = 1

        self._reset_graph_observables()

        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer()

        return self.get_observation()

    def _reset_graph_observables(self):
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action spins of value 0.
            self.matrix_obs = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[0] + 1))
            self.matrix_obs[:-1, :-1] = self.matrix
        else:
            self.matrix_obs = self.matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action spins of value 0.
                self.bias_obs = np.concatenate((self.bias, [0]))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, spins=None):
        state = np.zeros((self.observation_space.shape[1], self.n_actions))
        if spins is None:
            if self.reversible_spins:
                if self.spin_basis == SpinBasis.BINARY:
                    state[0, :self.n_spins] = np.random.randint(2, size=self.n_spins)
                else:
                    state[0, :self.n_spins] = 2 * np.random.randint(2, size=self.n_spins) - 1
            else:
                if self.spin_basis == SpinBasis.BINARY:
                    state[0, :self.n_spins] = 1  # Context dependent initial state
                else:
                    state[0, :self.n_spins] = 1
        else:
            if self.spin_basis == SpinBasis.BINARY:
                if not np.isin(spins, [0, 1]).all():
                    raise Exception("SpinSystem is configured for binary spins ([0,1]).")
                state[0, :self.n_spins] = spins
            else:
                state[0, :self.n_spins] = self._format_spins_to_signed(spins)

        state = state.astype('float')

        # Observables Update
        self._update_observables(state)
        return state

    def _update_observables(self, state):
        spins = state[0, :self.n_spins]
        immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable(spins=spins)

        for idx, obs in self.observables:
            if obs == Observable.IMMEDIATE_REWARD_AVAILABLE:
                state[idx, :self.n_spins] = immeditate_rewards_avaialable / self.max_local_reward_available
            elif obs == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                state[idx, :self.n_spins] = 1 - np.sum(immeditate_rewards_avaialable <= 0) / self.n_spins
            elif obs == Observable.NEIGHBOR_COVERAGE:
                if self.optimisation_target == OptimisationTarget.MTDS:
                    coverage = get_neighbor_coverage(self.matrix, spins, self.spin_basis)
                    max_degree = np.max(np.sum(self.matrix != 0, axis=1))
                    state[idx, :self.n_spins] = coverage / max_degree if max_degree > 0 else 0
                    self._initial_coverage = coverage.copy()
                else:
                    state[idx, :self.n_spins] = 0
            elif obs == Observable.TDS_VALIDITY_RATIO:
                if self.optimisation_target == OptimisationTarget.MTDS:
                    validity = get_tds_validity_ratio(self.matrix, spins, self.spin_basis)
                    state[idx, :] = validity
                    self._initial_validity = validity
                else:
                    state[idx, :] = 1.0

    def _get_spins(self, basis=None):
        if basis is None:
            basis = self.spin_basis
        spins = self.state[0, :self.n_spins].copy()

        if basis == SpinBasis.SIGNED:
            pass
        elif basis == SpinBasis.BINARY:
            # If current state is signed but requested binary is complicated if state is mixed
            # Assuming state is consistent with self.spin_basis
            if self.spin_basis == SpinBasis.SIGNED:
                 spins = (1 - spins) / 2
        return spins

    def calculate_best_energy(self):
        # ... (Same as original, assuming logic is correct for Brute Force) ...
        if self.n_spins <= 10:
            res = self.calculate_best_brute()
        else:
            n_cpu = int(mp.cpu_count()) / 2
            pool = mp.Pool(mp.cpu_count())
            iMax = 2 ** (self.n_spins)
            args = np.round(np.linspace(0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
            arg_pairs = [list(args) for args in zip(args, args[1:])]
            try:
                res = pool.starmap(self._calc_over_range, arg_pairs)
                idx_best = np.argmin([e for e, s in res])
                res = res[idx_best]
            except Exception:
                res = self._calc_over_range(0, 2 ** (self.n_spins))
            finally:
                pool.close()

            if self.spin_basis == SpinBasis.BINARY:
                best_score, best_spins = res
                best_spins = (1 - best_spins) / 2
                res = best_score, best_spins

            if self.optimisation_target == OptimisationTarget.DSP:
                best_energy, best_spins = res
                best_cut = self.calculate_dsp(best_spins)
                res = best_cut, best_spins
            else:
                # For MTDS brute force calculation if needed
                pass
        return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    # Note: step() is fully overridden in Unbiased class now for MTDS
    @abstractmethod
    def step(self, action, random_weight, adj_mask=False):
        pass

    def get_observation(self):
        state = self.state.copy()

        if self.spin_basis == SpinBasis.BINARY:
            state[0, :] = (1 - state[0, :]) / 2

        if self.gg.biased:
            return np.vstack((state, self.matrix_obs, self.bias_obs))
        else:
            return np.vstack((state, self.matrix_obs))

    def _get_selected_size(self):
        if self.spin_basis == SpinBasis.BINARY:
            return int(np.sum(self.state[0, :self.n_spins] == 0))
        else:
            return int(np.sum(self.state[0, :self.n_spins] == -1))


    def get_immeditate_rewards_avaialable(self, spins=None):
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target == OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1 * self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target == OptimisationTarget.DSP:
            immediate_reward_function = self._get_immeditate_dsps_avaialable_jit
        elif self.optimisation_target == OptimisationTarget.MTDS:
            immediate_reward_function = self._get_immeditate_mtds_rewards_available
        else:
            raise NotImplementedError("Optimisation target {} not recognised.".format(self.optimisation_ta))

        spins = spins.astype('float64')
        matrix = self.matrix_obs.astype('float64')
        if self.gg.biased:
            bias = self.bias.astype('float64')
            if self.optimisation_target == OptimisationTarget.MTDS:
                return immediate_reward_function(spins)  # MTDS不需要矩阵
            else:
                return immediate_reward_function(spins, matrix, bias)
        else:
            if self.optimisation_target == OptimisationTarget.MTDS:
                return immediate_reward_function(spins)
            else:
                return immediate_reward_function(spins, matrix)

    def get_allowed_action_states(self):
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0, 1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (1, -1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis == SpinBasis.BINARY:
                return 0
            if self.spin_basis == SpinBasis.SIGNED:
                return 1

    def calculate_score(self, spins=None):
        """
            Base score calculation. Overridden logic handled in subclasses or specific targets.
        """
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target == OptimisationTarget.DSP:
            score = self.calculate_dsp(spins)
        elif self.optimisation_target == OptimisationTarget.MTDS:
            if self.spin_basis == SpinBasis.BINARY:
                score = int(np.sum(spins == 0))
            else:
                score = int(np.sum(spins == -1))
        else:
            raise NotImplementedError

        return score

    def _calculate_score_change(self, new_spins, matrix, action, random_weight, ifweight):
        if self.optimisation_target == OptimisationTarget.DSP:
            delta_score = self._calculate_dsp_change(
                new_spins, matrix, action, random_weight, ifweight
            )
        elif self.optimisation_target == OptimisationTarget.MTDS:
            delta_score = self._calculate_mtds_change(
                new_spins, matrix, action, random_weight, ifweight
            )
        else:
            raise NotImplementedError

        return delta_score

    def _format_spins_to_signed(self, spins):
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception("SpinSystem is configured for binary spins ([0,1]).")
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception("SpinSystem is configured for signed spins ([-1,1]).")
        return spins

    @abstractmethod
    def calculate_mtds(self, spins=None):
        """calculate MTDS size"""
        raise NotImplementedError

    @abstractmethod
    def get_best_mtds(self):
        """get the best MTDS size"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_mtds_change(self, new_spins, matrix, action, random_weight, ifweight):
        """calculate MTDS score change"""
        raise NotImplementedError

    @abstractmethod
    def _get_immeditate_mtds_rewards_available(self, spins):
        """get the immeditate MTDS rewards available"""
        raise NotImplementedError

    @abstractmethod
    def calculate_dsp(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_dsp(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    # @abstractmethod
    # def _calculate_energy_change(self, new_spins, matrix, action):
    #     raise NotImplementedError

    @abstractmethod
    def _calculate_dsp_change(self, new_spins, matrix, action, random_weight, ifweight):
        raise NotImplementedError


##########
# Classes for implementing the calculation methods with/without biases.
##########
class SpinSystemUnbiased(SpinSystemBase):
    # 1. 辅助函数：执行动作
    def _apply_action(self, action):
        """
        执行动作逻辑：翻转节点状态（支持可逆操作）
        """
        # 处理 Extra Action (如 Pass 或 Randomise)
        if action == self.n_spins:
            if self.extra_action == ExtraAction.PASS:
                pass
            elif self.extra_action == ExtraAction.RANDOMISE:
                if self.spin_basis == SpinBasis.BINARY:
                    self.state[0, :self.n_spins] = np.random.randint(2, size=self.n_spins)
                else:
                    random_actions = np.random.choice([1, -1], self.n_spins)
                    self.state[0, :self.n_spins] = self.state[0, :self.n_spins] * random_actions
            return

        # 常规动作：翻转节点 (因为 reversible_spins=True，所以取反)
        if self.spin_basis == SpinBasis.BINARY:
            self.state[0, action] = 1 - self.state[0, action]
        else:
            self.state[0, action] = -self.state[0, action]

    def step(self, action, random_weight=None, adj_mask=False):
        self.current_step += 1
        if self.optimisation_target != OptimisationTarget.MTDS:
            raise NotImplementedError("This customized SpinSystem only supports MTDS.")

        old_state = self.state.copy()
        old_validity_ratio = get_tds_validity_ratio(
            self.matrix, old_state[0, :self.n_spins], self.spin_basis
        )
        if self.spin_basis == SpinBasis.BINARY:
            old_size = np.sum(old_state[0, :self.n_spins] == 0)
        else:
            old_size = np.sum(old_state[0, :self.n_spins] == -1)

        self._apply_action(action)

        validity_ratio = get_tds_validity_ratio(
            self.matrix,
            self.state[0, :self.n_spins],
            self.spin_basis
        )

        if self.spin_basis == SpinBasis.BINARY:
            current_size = np.sum(self.state[0, :self.n_spins] == 0)
        else:
            current_size = np.sum(self.state[0, :self.n_spins] == -1)

        size_change = old_size - current_size
        validity_change = validity_ratio - old_validity_ratio
        normalized_size = current_size / self.n_spins

        rew = 0.0

        P = getattr(self, 'mtds_constraint_penalty', 3.0)

        if validity_ratio < 1.0:
            rew -= 0.2
            rew -= 1.0 * P * (1.0 - validity_ratio)

            if validity_change > 0:
                rew += validity_change * P * 2.0
            elif validity_change < 0:
                rew -= 0.5 * P

            if size_change > 0 and validity_change < 0:
                rew -= 0.5 * P

        else:
            if old_validity_ratio < 1.0:
                rew += 1.0 * P

            rew -= 0.05 * normalized_size

            if size_change > 0:
                base_bonus = 0.5
                rew += base_bonus + 1.0 * (1.0 - normalized_size)

            elif size_change < 0:
                rew -= 0.5

        rew -= 0.01

        done = False
        if self.current_step >= self.max_steps:
            done = True

        self.score = current_size

        if validity_ratio >= 1.0:
            if current_size < self.best_score or self.best_score == np.inf:
                self.best_score = current_size
                self.best_spins = self.state[0, :self.n_spins].copy()

        # Update Observables
        self._update_observables(self.state)

        for idx, observable in self.observables:
            if observable == Observable.TIME_SINCE_FLIP:
                self.state[idx, :] += (1. / self.max_steps)
                if action < self.n_spins:
                    self.state[idx, action] = 0
            elif observable == Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)
            elif observable == Observable.TERMINATION_IMMANENCY:
                self.state[idx, :] = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

        if self.memory_length is not None:
            self.score_memory[self.idx_memory] = self.score
            self.spins_memory[self.idx_memory] = self.state[0, :self.n_spins]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = np.min(self.score_memory)

        info = {
            "validity_ratio": validity_ratio,
            "current_size": current_size,
        }
        return self.get_observation(), rew, done, info

    def apply_mtds_remedy(self, spins=None):
        """
        MTDS Remedy Mechanism (Algorithm 2)
        """
        if spins is None:
            spins = self.state[0, : self.n_spins].copy()
        else:
            spins = np.copy(spins)

        n = self.n_spins
        adj = self.matrix

        # 识别支配集S中的顶点
        if self.spin_basis == SpinBasis.BINARY:
            S = set(np.where(spins == 0)[0])
        else:
            S = set(np.where(spins == -1)[0])

        changed = False
        # 动态调整迭代次数，至少10次，或者 log2(n)
        max_iterations = max(10, int(np.log2(n) * 2))

        for iteration in range(max_iterations):
            old_S = S.copy()

            # ===== Phase 1: 修复支配性 =====
            # 检查是否所有顶点都被S支配
            undominated = []
            for v in range(n):
                neighbors = set(np.where(adj[v, :] != 0)[0])
                neighbors.discard(v)
                if not (neighbors & S):  # 邻接与S交集为空
                    undominated.append(v)

            for v in undominated:
                neighbors = set(np.where(adj[v, :] != 0)[0])
                neighbors.discard(v)
                if len(neighbors) > 0:
                    # 选择度数最大的邻居加入S
                    best_neighbor = max(neighbors,
                                        key=lambda x: np.sum(adj[x, :] != 0))
                    S.add(best_neighbor)
                    changed = True

            # ===== Phase 2: 修复全总性 =====
            # 检查S中的每个顶点是否都被S支配
            for u in list(S):
                neighbors = set(np.where(adj[u, :] != 0)[0])
                neighbors.discard(u)
                if not (neighbors & S):  # u在S中但是孤立
                    if len(neighbors) > 0:
                        # 选择度数最大的邻居加入S
                        best_neighbor = max(neighbors,
                                            key=lambda x: np.sum(adj[x, :] != 0))
                        S.add(best_neighbor)
                        changed = True

            # 检查是否收敛
            if S == old_S:
                break

        # 转换回自旋表示
        if self.spin_basis == SpinBasis.BINARY:
            new_spins = np.ones(n)
            new_spins[list(S)] = 0
        else:
            new_spins = np.ones(n)
            new_spins[list(S)] = -1

        return new_spins, changed

    def calculate_energy(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(spins, matrix)

    def calculate_mtds(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        if self.spin_basis == SpinBasis.BINARY:
            return int(np.sum(spins == 0))
        else:
            return int(np.sum(spins == -1))

    def calculate_dsp(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            if self.spin_basis == SpinBasis.BINARY:
                if not np.isin(spins, [0, 1]).all():
                    raise Exception("SpinSystem is configured for binary spins ([0,1]).")
                spins = 2 * spins - 1  # 转换为[-1, 1]
            elif self.spin_basis == SpinBasis.SIGNED:
                if not np.isin(spins, [-1, 1]).all():
                    raise Exception("SpinSystem is configured for signed spins ([-1,1]).")

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return (1 / 4) * np.sum(np.multiply(matrix, 1 - np.outer(spins, spins)))

    def get_best_mtds(self):
        """
        返回历史最优的 MTDS 大小
        """
        if self.optimisation_target == OptimisationTarget.MTDS:
            return self.calculate_mtds(self.best_spins)
        else:
            raise NotImplementedError("Can't return best MTDS when optimisation target is not MTDS")

    def _calculate_mtds_change(self, new_spins, matrix, action, random_weight, ifweight):
        if self.spin_basis == SpinBasis.BINARY:
            action_in_dominating_set = (new_spins[action] == 0)
        else:
            action_in_dominating_set = (new_spins[action] == -1)

        degree = np.sum(matrix[: , action] != 0) - (matrix[action, action] != 0)

        neighbors = np.where(matrix[action, : ] != 0)[0]
        neighbors = neighbors[neighbors != action]

        if self.spin_basis == SpinBasis.BINARY:
            covered_neighbors = np.sum([1 for nb in neighbors if new_spins[nb] == 0])
        else:
            covered_neighbors = np.sum([1 for nb in neighbors if new_spins[nb] == -1])

        uncovered_neighbors = len(neighbors) - covered_neighbors

        if ifweight:
            spins_state = np.copy(new_spins).astype(float)
            if self.spin_basis == SpinBasis.SIGNED:
                spins_state[spins_state == -1] = 0
            delta = np.sum((spins_state * matrix[: , action]) * random_weight) + random_weight[action]
        else:
            if action_in_dominating_set:
                delta = -degree
            else:
                delta = degree
                if uncovered_neighbors > 0:
                    delta *= 1.5
                elif uncovered_neighbors == 0:
                    delta *= 0.5

        return delta

    def _get_immeditate_mtds_rewards_available(self, spins, matrix=None):
        n = self.n_spins

        # 兼容性处理
        if matrix is None:
            if hasattr(self, 'matrix_obs'):
                matrix = self.matrix_obs
            elif hasattr(self, 'matrix'):
                matrix = self.matrix
            else:
                raise ValueError("Matrix is required for MTDS")

        if self.immediate_reward_mode == "RL":
            return np.zeros(n)

        elif self.immediate_reward_mode == "GREEDY":
            # 1. 确定当前选中状态 S
            # Signed Basis: -1 is Selected
            if self.spin_basis == SpinBasis.BINARY:
                is_selected = (spins <= 0.1)
            else:
                is_selected = (spins <= -0.1)

            # 2. 计算当前的支配状态 (TDS 定义)
            # 每个节点被多少个 S 中的邻居支配？
            # current_dom_counts[i] = (i 的邻居中有多少个在 S 中)
            current_dom_counts = matrix.dot(is_selected.astype(float))

            # 3. 找出当前"未被覆盖"的节点 (Uncovered)
            # MTDS 定义：只要邻居里没有 S 的人，就算未覆盖 (dom_counts == 0)
            # 即使 i 自己在 S 里，如果它没有邻居在 S 里，它依然是 Uncovered！
            uncovered_mask = (current_dom_counts == 0)

            # === 终止条件 ===
            # 如果所有节点都有了邻居支配 (uncovered_mask 全 False)，说明已合法，停止
            if not np.any(uncovered_mask):
                return np.full(n, -np.inf)  # 或者是 0，配合 Greedy.step 的停止逻辑

            rewards = np.zeros(n)

            # === 已经在 S 中的点 ===
            # Constructive Greedy 不允许移除
            rewards[is_selected] = -np.inf

            # === 不在 S 中的点 (Candidates) ===
            # 计算增益：如果加入节点 i，它能覆盖谁？
            # 答：它能覆盖它的所有邻居 N(i)。
            # Gain = N(i) 中有多少个是当前 Uncovered 的。

            candidate_mask = ~is_selected

            # 矩阵向量乘法：Gain[i] = sum(uncovered_mask[j] for j in neighbors(i))
            gains = matrix.dot(uncovered_mask.astype(float))

            # 【核心区别】
            # 在 TDS 中，加入 i 并不能覆盖 i 自己。
            # 所以这里 *不* 加 self_gain。

            rewards[candidate_mask] = gains[candidate_mask]

            return rewards

        else:
            return np.zeros(n)

    def get_best_dsp(self):
        if self.optimisation_target == OptimisationTarget.MTDS:
            return self.get_best_mtds()
        elif self.optimisation_target == OptimisationTarget.DSP:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best DSP solve when optimisation target is set to energy.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix)

    def _calculate_dsp_change(self, new_spins, matrix, action, random_weight, ifweight):
        if ifweight:
            spins_state = deepcopy(new_spins)
            spins_state[spins_state == -1] = 0
            res = np.sum((spins_state * matrix[:, action]) * random_weight) + random_weight[action]
            return res
        else:
            return matmul(new_spins.T, matrix[:, action])  # new_spins[action] *


    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - matmul(spins.T, matmul(matrix, spins)) / 2
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins


    @staticmethod
    def _get_immeditate_dsps_avaialable_jit(spins, matrix):
        spins_state = copy(spins)
        spins_state[spins_state == -1] = 0
        return matmul(matrix, spins)


class SpinSystemBiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if type(spins) == type(None):
            spins = self._get_spins()

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')

        return self._calculate_energy_jit(spins, matrix, bias)

    def calculate_mtds(self, spins=None):
        raise NotImplementedError("MTDS not defined for biased SpinSystems")

    def get_best_mtds(self):
        raise NotImplementedError("MTDS not defined for biased SpinSystems")

    def _calculate_mtds_change(self, new_spins, matrix, action, random_weight, ifweight):
        raise NotImplementedError("MTDS not defined for biased SpinSystems")

    def _get_immeditate_mtds_rewards_available(self, spins):
        raise NotImplementedError("MTDS not defined for biased SpinSystems")

    def calculate_dsp(self, spins=None):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    def get_best_dsp(self):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix, bias)

    @staticmethod
    @jit(nopython=True)
    def _calculate_dsp_change(new_spins, matrix, bias, action):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_jit(spins, matrix, bias):
        return matmul(spins.T, matmul(matrix, spins)) / 2 + matmul(spins.T, bias)

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix, bias):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            current_energy = -(matmul(spins.T, matmul(matrix, spins)) / 2 + matmul(spins.T, bias))
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix, bias):
        return - (2 * spins * (matmul(matrix, spins) + bias))

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_dsps_avaialable_jit(spins, matrix, bias):
        raise NotImplementedError("DSP not defined/implemented for biased SpinSystems.")

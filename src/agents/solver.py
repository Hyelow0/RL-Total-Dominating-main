from abc import ABC, abstractmethod

import numpy as np
import torch

class SpinSolver(ABC):
    """Abstract base class for agents solving SpinSystem Ising problems."""

    def __init__(self, env, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        """Base initialisation of a SpinSolver.

        Args:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool, optional): The logging verbosity.

        Attributes:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool): The logging verbosity.
            total_reward (float): The cumulative total reward received.
        """

        self.env = env
        self.verbose = verbose
        self.record_cut = record_cut
        self.record_rewards = record_rewards
        self.record_qs = record_qs

        self.total_reward = 0

    def reset(self):
        self.total_reward = 0
        self.env.reset()

    def solve(self, *args, random_weight):
        """Solve the SpinSystem by flipping individual spins until termination.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Returns:
            (float): The cumulative total reward received.

        """
        action_list = []
        done = False
        while not done:
            reward, done, action = self.step(*args, random_weight)
            action_list.append(action)
            self.total_reward += reward
        return self.total_reward, action_list

    @abstractmethod
    def step(self, *args, random_weight):
        """Take the next step (flip the next spin).

        The implementation of 'step' depedens on the
                solver instance used.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Raises:
            NotImplementedError: Every subclass of SpinSolver must implement the
                step method.
        """

        raise NotImplementedError()

class Greedy(SpinSolver):
    """A greedy solver for a SpinSystem."""

    def __init__(self, *args, **kwargs):
        """Initialise a greedy solver.

        Args:
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            trial_env (SpinSystemMCTS): The environment with in the agent tests
                actions (a clone of self.env where the final actions are taken).
            current_snap: The current state of the environment.
        """

        super().__init__(*args, **kwargs)
        if hasattr(self.env, 'immediate_reward_mode'):
            self.env.immediate_reward_mode = "GREEDY"

    def step(self, random_weight=None):
        """Take the action which maximises the immediate reward."""

        # 1. 获取奖励向量 (MTDS Constructive Greedy: 奖励 = 新增覆盖数)
        rewards_available = self.env.get_immeditate_rewards_avaialable()

        if self.env.reversible_spins:
            # 可逆自旋逻辑 (通常 MTDS Constructive 不用这个，但保留)
            if np.max(rewards_available) <= 0:
                return 0, True, None
            action = rewards_available.argmax()
        else:
            # 不可逆自旋逻辑 (Constructive Greedy)
            masked_rewards = rewards_available.copy()

            # --- [修复 3] 获取 Observation 并处理维度 ---
            obs = self.env.get_observation()
            # 兼容 (1, N) 和 (N,) 两种形状
            current_spins = obs[0, :] if obs.ndim > 1 else obs

            # --- [修复 2] 使用 -inf 进行 Mask ---
            # 屏蔽掉非法的动作 (即: 已经是 -1 的点不能再选)
            np.putmask(masked_rewards,
                       current_spins != self.env.get_allowed_action_states(),
                       -np.inf)

            # --- [修复 1 & 关键] 刹车逻辑 (Stopping Condition) ---
            # 如果 Mask 后的最大奖励 <= 0，说明：
            # 1. 要么没点可选了 (全是 -inf)
            # 2. 要么剩下的点增益全是 0 (图已经完全覆盖了！)
            # 这时候必须立刻停止！
            max_reward = np.max(masked_rewards)

            if max_reward <= 0:
                # 停止：reward=0, done=True, action=None
                return 0, True, None

            # 只有在有正向收益时，才选择动作
            action = masked_rewards.argmax()

        # 执行动作
        observation, reward, done, _ = self.env.step(action, random_weight, adj_mask=False)

        return reward, done, action

class Random(SpinSolver):
    """A random solver for a SpinSystem."""

    def step(self, random_weight):
        """Take a random action.

        Returns:
            reward (float): The reward recieved.
            done (bool): Whether the environment is in a terminal state after
                the action is taken.
        """

        observation, reward, done, _ = self.env.step(self.env.action_space.sample(), random_weight, adj_mask=False)
        return reward, done

class Network(SpinSolver):
    """A network-only solver for a SpinSystem."""

    epsilon = 0.

    def __init__(self, network, *args, **kwargs):
        """Initialise a network-only solver.

        Args:
            network: The network.
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            current_snap: The last observation of the environment, used to choose the next action.
        """

        super().__init__(*args, **kwargs)
        if hasattr(self.env, 'immediate_reward_mode'):
            self.env.immediate_reward_mode = "RL"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.network.eval()
        self.current_observation = self.env.get_observation()
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)

        self.history = []

    def reset(self, spins=None, clear_history=True, reset_weight=True):
        if spins is None:
            self.current_observation = self.env.reset(reset_weight=reset_weight)
        else:
            self.current_observation = self.env.reset(spins)
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)
        self.total_reward = 0

        if clear_history:
            self.history = []

    @torch.no_grad()
    def step(self, random_weight=None):  # <--- 【修复 1】接收 random_weight 参数
        # Q-values predicted by the network.
        qs = self.network(self.current_observation)

        if self.env.reversible_spins:
            if np.random.uniform(0, 1) >= self.epsilon:
                # Action that maximises Q function
                action = qs.argmax().item()
            else:
                # Random action
                action = np.random.randint(0, self.env.action_space.n)

        else:
            # 这里的逻辑是针对不可逆自旋的掩码处理
            x = (self.current_observation[0, :] == self.env.get_allowed_action_states()).nonzero()
            if x.numel() == 0:
                # 如果没动作可选，直接结束
                return 0, True, None # reward=0, done=True
            if np.random.uniform(0, 1) >= self.epsilon:
                action = x[qs[x].argmax().item()].item()
                # Allowed action that maximises Q function
            else:
                # Random allowed action
                action = x[np.random.randint(0, len(x))].item()

        if action is not None:
            # 【修复 2】使用传入的 random_weight，确保与 Greedy 面对相同的难题
            # 如果没传 (None)，再兜底使用 self.env.random_weight
            weight_to_use = random_weight if random_weight is not None else self.env.random_weight
            observation, reward, done, _ = self.env.step(action, weight_to_use, adj_mask=False)
            self.current_observation = torch.FloatTensor(observation).to(self.device)

        else:
            reward = 0
            done = True

        if not self.record_cut and not self.record_rewards:
            record = [action]
        else:
            record = [action]
            if self.record_cut:
                record += [self.env.calculate_cut()]
            if self.record_rewards:
                record += [reward]
            if self.record_qs:
                record += [qs]

        # record_stage1_new += [self.env.get_immeditate_rewards_avaialable()] # 这行可能会报错如果环境没实现，暂时注释或保留视情况而定

        self.history.append(record)

        # 【注意】SpinSolver.solve 期望返回 (reward, done, action)
        return reward, done, action

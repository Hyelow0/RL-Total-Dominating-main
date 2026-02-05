import argparse
import os
import sys
import pickle
from src.envs.utils import OptimisationTarget
from torch.utils.tensorboard import SummaryWriter
# 获取当前文件（run.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（RL-Dominating-main）
project_root = os.path.dirname(current_dir)
# 添加到 Python 路径
sys.path.insert(0, project_root)
import matplotlib.pyplot as plt
import numpy as np

import src.envs.core as ising_env
from experiments.utils import load_graph_set, mk_dir
from src.agents.dqn.dqn import DQN
from src.agents.dqn.utils import TestMetric
from src.envs.utils import (SetGraphGenerator,
                            RandomNonIsolatedGraphGenerator,
                            RandomBarabasiAlbertGraphGenerator,
                            EdgeType, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            Observable, RandomGraphGenerator, RandomErdosRenyiGraphGenerator)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns

    plt.style.use('seaborn')
except ImportError:
    pass

import time


def tb_plot(data_x, data_y, writer, img_name):
    if img_name is None:
        raise Exception('no img_name of plot curve!')
    else:
        for x, y in zip(data_x, data_y):
            writer.add_scalars(img_name, {'scores': y}, x)


def run(timesteps, n_spins, save_loc, args: argparse.Namespace, test_loc=None, verbose=True, optimisation_target=OptimisationTarget.MTDS, graph_type='NI'):
    """
    train function: support DSP and MTDS
    Args:
        optimisation_target: 优化目标
    """
    step_fact = 2
    env_args = {'observables': [
                    Observable.SPIN_STATE,
                    Observable.TIME_SINCE_FLIP,
                    Observable.NEIGHBOR_COVERAGE,       # MTDS新增
                    Observable.TDS_VALIDITY_RATIO       # MTDS新增
                ],
                'reward_signal': RewardSignal.DENSE,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': optimisation_target,
                'mtds_constraint_penalty': args.mtds_constraint_penalty,
                'spin_basis': SpinBasis.SIGNED,
                'norm_rewards': args.norm_reward,
                'memory_length': 50,
                'horizon_length': None,
                'stag_punishment': 1.0,
                'basin_reward': None,
                'reversible_spins': False,
                'ifweight': args.if_weight}

    n_spins_train = n_spins

    train_graph_generator = RandomNonIsolatedGraphGenerator(n_spins=n_spins_train, edge_type=EdgeType.DISCRETE)
    # train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection= 0.4, edge_type=EdgeType.DISCRETE)

    if args.grid:
        graph_save_loc = test_loc
        graphs_test = load_graph_set(graph_save_loc)
        test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)
    else:
        test_graph_generator = RandomNonIsolatedGraphGenerator(n_spins=n_spins_train, edge_type=EdgeType.DISCRETE)
        # test_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection=0.4, edge_type=EdgeType.DISCRETE)



    train_envs = [ising_env.make("SpinSystem",
                                 train_graph_generator,
                                 int(n_spins_train * step_fact),
                                 **env_args)]

    n_spins_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("SpinSystem",
                                test_graph_generator,
                                int(n_spins_test * step_fact),
                                **env_args)]

    # network_save_path = save_loc + 'network.pth'
    # network_save_path_batch = [network_save_path.replace('20', str(i)) for i in
    #                            [20, 40, 80, 100, 200, 300, 400, 500, 800]]
    # test_save_path = save_loc + 'test_scores.pkl'
    # loss_save_path = save_loc + 'losses.pkl'

    if verbose:
        print(f"[INFO] Manually injecting mtds_constraint_penalty = {args.mtds_constraint_penalty}")
    for env in train_envs:
        env.mtds_constraint_penalty = args.mtds_constraint_penalty
    for env in test_envs:
        env.mtds_constraint_penalty = args.mtds_constraint_penalty
    # 改进后的跨节点保存路径构造
    base_dir = os.path.dirname(os.path.dirname(save_loc))  # 'checkpoints/BA_graphs/'

    network_save_path_batch = []
    for target_spins in [20, 40, 80, 100, 200, 300, 400, 500]:
        target_dir = os.path.join(base_dir, f'{target_spins}spins')

        # 为所有目录创建文件夹
        os.makedirs(target_dir, exist_ok=True)

        if target_spins == n_spins:
            # 当前训练的节点目录，同时保存两个文件

            # 1. 标准名称（network.pth）
            standard_file = os.path.join(target_dir, 'network.pth')
            network_save_path_batch.append(standard_file)

            # 2. 标注名称（network_{n_spins}.pth）
            annotated_file = os.path.join(target_dir, f'network_{n_spins}.pth')
            network_save_path_batch.append(annotated_file)

            if verbose:
                print(f"当前节点 {target_spins}spins 保存路径:")
                print(f"   1. {standard_file}")
                print(f"   2. {annotated_file}")
        else:
            # 其他节点目录，保存为带标注的名称
            target_file = os.path.join(target_dir, f'network_{n_spins}.pth')
            network_save_path_batch.append(target_file)

            if verbose:
                print(f"   跨节点保存路径 {target_spins}spins:   {target_file}")


    network_save_path = save_loc + 'network.pth'
    test_save_path = save_loc + 'test_scores.pkl'
    loss_save_path = save_loc + 'losses.pkl'

    nb_steps = timesteps

    network_fn = lambda: MPNN(n_obs_in=train_envs[0].observation_space.shape[1],
                              n_layers=3,
                              n_features=64,
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(train_envs,
                network_fn,

                init_network_params=None, # '../checkpoints/NI_graphs/20spins/network_best_20.pth'
                init_weight_std=args.init_weight_std,

                double_dqn=True,
                clip_Q_targets=True,
                stage_two_mode=False,

                replay_start_size=500,
                replay_buffer_size=args.replay_buffer_size,  # 20000
                gamma=args.gamma,  # 1
                update_target_frequency=args.update_target_frequency,  # 5000

                update_learning_rate=True,#False True
                initial_learning_rate=5e-5,
                peak_learning_rate=args.lr,#1e-4
                peak_learning_rate_step=10000,#20000
                final_learning_rate=1e-6,#1e-5
                final_learning_rate_step=600000,#600000

                update_frequency=1,  # 1,32
                minibatch_size=args.minibatch_size,  # 128
                max_grad_norm=None,#None
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=args.initial_exploration_rate,
                final_exploration_rate=args.final_exploration_rate,  # 0.05
                final_exploration_step=args.final_exploration_step,  # 40000

                adam_epsilon=1e-8,
                logging=True,
                loss="mse",

                save_network_frequency=100000,
                network_save_path=network_save_path,
                network_save_path_batch=network_save_path_batch,

                evaluate=True,
                test_grid_graph=False,
                test_envs=test_envs,
                test_envs_batch=20,
                test_episodes=20,
                test_frequency=10000,  # 10000
                test_save_path=test_save_path,
                test_metric=TestMetric.TDS_SIZE if optimisation_target == OptimisationTarget.MTDS else TestMetric.MAX_CUT,
                randweight=args.if_weight,
                seed=None,
                verbose=verbose
                )
    if verbose:
        print("\n Created DQN agent with network:\n\n", agent.network)

    start = time.time()
    scores = agent.learn(timesteps=nb_steps)
    if verbose:
        print('Cost time: ', time.time() - start)
    agent.save()
    if verbose:
        plt.clf()
        ############
        # PLOT - learning curve
        ############
        data = pickle.load(open(test_save_path, 'rb'))
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel("Timestep")
        plt.ylabel("Mean reward")
        if agent.test_metric == TestMetric.ENERGY_ERROR:
            plt.ylabel("Energy Error")
        elif agent.test_metric == TestMetric.BEST_ENERGY:
            plt.ylabel("Best Energy")
        elif agent.test_metric == TestMetric.CUMULATIVE_REWARD:
            plt.ylabel("Cumulative Reward")
        elif agent.test_metric == TestMetric.MAX_CUT:
            plt.ylabel("Dominating Set Scores")
        elif agent.test_metric == TestMetric.FINAL_CUT:
            plt.ylabel("Final Cut")
        elif agent.test_metric == TestMetric.TDS_SIZE:
            plt.ylabel("MTDS Size (Lower is Better)")

        plt.savefig(save_loc + "training_curve.png", bbox_inches='tight', dpi=300)
        plt.savefig(save_loc + "training_curve.pdf", bbox_inches='tight', dpi=300)
        plt.clf()

        ############
        # PLOT - losses
        ############
        if os.path.exists(loss_save_path):
            data = pickle.load(open(loss_save_path, 'rb'))
            data = np.array(data)
            if len(data) > 0:
                N = 50
                if len(data) > N:
                    data_x = np.convolve(data[:, 0], np.ones((N,)) / N, mode='valid')
                    data_y = np.convolve(data[:, 1], np.ones((N,)) / N, mode='valid')
                    plt.plot(data_x, data_y)
                else:
                    plt.plot(data[:, 0], data[:, 1])

                plt.xlabel("Timestep")
                plt.ylabel("Loss")

                plt.yscale("log")
                plt.grid(True)
                plt.savefig(save_loc + "loss.png", bbox_inches='tight', dpi=300)
                plt.savefig(save_loc + "loss.pdf", bbox_inches='tight', dpi=300)
    return scores

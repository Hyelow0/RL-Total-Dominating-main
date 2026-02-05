import argparse
import os
import re

import matplotlib.pyplot as plt
import torch

import src.envs.core as ising_env
from experiments.utils import test_network, load_graph_set
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            Observable, EdgeType, GraphGenerator)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns

    plt.style.use('seaborn')
except ImportError:
    pass


def run(save_loc, graph_save_loc, args: argparse.Namespace, batched=True,
        max_batch_size=None, p=None, m=None, model_name=None, optimisation_target=OptimisationTarget.MTDS, custom_model_path=None):
    """
    测试函数，支持MTDS
    """
    # 第1步：从 save_loc 中提取当前节点数
    match = re.search(r'(\d+)spins', save_loc)
    if match:
        current_spins = int(match.group(1))
        # print(f"当前测试节点数:   {current_spins}")
    else:
        raise ValueError(f"Unable to extract the number of nodes from the path: {save_loc}")

    # 第2步：确定加载哪个模型文件
    if custom_model_path is not None:
        # 如果用户强行指定了路径，直接使用，无视其他逻辑
        network_save_path = custom_model_path
        print(f"[INFO] Using a custom cross-origin model: {network_save_path}")
        if not os.path.exists(network_save_path):
            raise FileNotFoundError(f"The specified custom model file does not exist: {network_save_path}")
    elif model_name is not None:
        # 用户指定了模型
        network_save_path = save_loc + 'network_best_' + model_name + '.pth'
        print(f"Use the user-specified model:  network_best_{model_name}.pth")
    else:
        # 默认：优先使用当前节点训练的模型
        default_model_path = save_loc + f'network_best_{current_spins}.pth'
        fallback_model_path = save_loc + 'network_best_20.pth'

        if os.path.exists(default_model_path):
            network_save_path = default_model_path
        elif os.path.exists(fallback_model_path):
            network_save_path = fallback_model_path
        else:
            # 这里保留原有报错逻辑，或者你可以改成在这里 print warning
            raise FileNotFoundError(f"The default model file was not found in {save_loc}!")

    # 第3步：验证模型文件存在
    if not os.path.exists(network_save_path):
        raise FileNotFoundError(
            f"Model file does not exist: {network_save_path}\n"
            f"   Files in the directory: {os.listdir(save_loc)}"
        )

    print(f"The final loaded model: {network_save_path}\n")

    data_folder = os.path.join(save_loc, 'data')

    ####################################################
    # NETWORK SETUP
    ####################################################
    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }
    # 必须与 train.py 保持一致，或者是 train.py 的倍数。
    # 既然训练时给了 2.0 倍步数让它删点，测试时也要给够时间。
    step_factor = 2


    env_args = {'observables': [
                    Observable.SPIN_STATE,
                    Observable.TIME_SINCE_FLIP,
                    Observable.NEIGHBOR_COVERAGE,
                    Observable.TDS_VALIDITY_RATIO
                ],
                'reward_signal': RewardSignal.DENSE,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': optimisation_target,
                'mtds_constraint_penalty': getattr(args, 'mtds_constraint_penalty', 1.0),
                'spin_basis': SpinBasis.SIGNED,
                'norm_rewards': args.norm_reward,
                'memory_length': 50,
                'horizon_length': None,
                'stag_punishment': args.stag_punishment if args.stag_punishment is not None else 1.0,
                'basin_reward': None,
                'reversible_spins': True,
                'ifweight': args.if_weight}

    graphs_test = load_graph_set(graph_save_loc)

    test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(graphs_test[0]),
                              graphs_test[0].shape[0] * step_factor,
                              **env_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)

    # print(f"[DEBUG] 网络输入维度计算:")
    # print(f"  - 网络输入维度 (n_obs_in): {n_obs_in}\n")
    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    # 加载模型
    print(f"loading model: {network_save_path}")
    network.load_state_dict(torch.load(network_save_path, map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()
    print(f"loaded model!\n")

    results = test_network(network, env_args, graphs_test, device, step_factor,
                           return_raw=True, return_history=True,
                           batched=batched, max_batch_size=max_batch_size)

    filename_suffix = 'test_res'
    if custom_model_path is not None:
        # 如果是自定义模型，文件名加上 "_custom" 或模型来源标记
        filename_suffix = 'test_res_custom'
    elif model_name is not None:
        filename_suffix = f'test_res_{model_name}'

    # 保存测试结果
    if p is not None:
        # 例如: .../ER_p0.1_test_res_custom.xlsx
        prefix = save_loc + p + '_'
    elif m is not None:
        prefix = save_loc + m + '_'
    else:
        prefix = save_loc

    # 执行保存
    full_path_pkl = prefix + filename_suffix + '.pkl'
    full_path_xlsx = prefix + filename_suffix + '.xlsx'

    output_dir = os.path.dirname(full_path_pkl)
    if not os.path.exists(output_dir):
        print(f"[INFO] Directory doesn't exist, creating it: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    results.to_pickle(full_path_pkl)
    results.to_excel(full_path_xlsx)

    # # 保存测试结果
    # if p is not None:
    #     results.to_pickle(save_loc + p + 'test_res.pkl')
    #     results.to_excel(save_loc + p + 'test_res.xlsx')
    # elif m is not None:
    #     results.to_pickle(save_loc + m + 'test_res.pkl')
    #     results.to_excel(save_loc + m + 'test_res.xlsx')
    # else:
    #     results.to_pickle(save_loc + 'test_res.pkl')
    #     results.to_excel(save_loc + 'test_res.xlsx')
    print(f"The test results have been saved to: {full_path_xlsx}")

if __name__ == "__main__":
    run()

    # results_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    # results_raw_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    # history_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"

    # for res, fname, label in zip([results, results_raw, history],
    #                              [results_fname, results_raw_fname, history_fname],
    #                              ["results", "results_raw", "history"]):
    #     save_path = os.path.join(data_folder, fname)
    #     res.to_pickle(save_path)
        # print("{} saved to {}".format(label, save_path))

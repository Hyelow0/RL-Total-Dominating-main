import os
import pickle
import networkx as nx
import time
import numpy as np
import scipy as sp
import pandas as pd
import torch

from collections import namedtuple
from copy import deepcopy

import src.envs.core as ising_env
from src.envs.utils import (SingleGraphGenerator, SpinBasis, OptimisationTarget, calculate_mtds_size,
                            is_total_dominating_set, prune_solution_fast)
from src.agents.solver import Network, Greedy

####################################################
# Auxiliary function
####################################################
# Obtain the best score based on the optimization objective.
def get_best_score(env, optimisation_target):
    """
    Get the best score based on the optimisation target.
    Note: For MTDS, this calls get_best_mtds() which typically includes Remedy logic.
    """
    if optimisation_target == OptimisationTarget.MTDS:
        return env.get_best_mtds()
    elif optimisation_target == OptimisationTarget.DSP:
        return env.get_best_dsp()
    else:
        return env.best_energy

# check MTDS validity
def is_valid_mtds_check(nx_graph, node_set):
    """
    快速验证一个节点集合是否构成合法的 Total Dominating Set
    """
    # 转换为 set 提高查找速度
    node_set = set(node_set)
    for u in nx_graph.nodes():
        # 检查 u 的邻居中是否有被选中的点
        has_neighbor = False
        for v in nx_graph.neighbors(u):
            if v in node_set:
                has_neighbor = True
                break
        if not has_neighbor:
            return False
    return True

def should_minimize(optimisation_target):
    return optimisation_target == OptimisationTarget.MTDS

####################################################
# TESTING ON GRAPHS
####################################################
def test_network(network, env_args, graphs_test, device=None, step_factor=1, batched=True,
                 n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if batched:
        return __test_network_batched(network, env_args, graphs_test, device, step_factor,
                                      n_attempts, return_raw, return_history, max_batch_size)
    else:
        if max_batch_size is not None:
            print("Warning: max_batch_size argument will be ignored for when batched=False.")
        return __test_network_sequential(network, env_args, graphs_test, step_factor,
                                         n_attempts, return_raw, return_history)


def __test_network_batched(network, env_args, graphs_test, device=None, step_factor=1,
                           n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)

    # GET THE TARGET
    optimisation_target = env_args.get('optimisation_target', OptimisationTarget.DSP)
    is_minimization = should_minimize(optimisation_target)

    # HELPER FUNCTION FOR NETWORK TESTING
    acting_in_reversible_spin_env = env_args['reversible_spins']

    if env_args['reversible_spins']:
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = (0, 1)
        elif env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = (1, -1)
    else:
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = 0
        if env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = 1

    # 【辅助函数】只计算选中点的数量，不检查合法性，用于获取真实的 Raw Size
    def count_raw_size(spins, basis):
        if basis == SpinBasis.BINARY:
            return int(np.sum(spins == 0))
        else:
            return int(np.sum(spins == -1))

    # 确定目标值 (0 或 -1 代表选中)，用于 Validity Check
    target_val_def = 0 if env_args['spin_basis'] == SpinBasis.BINARY else -1

    def predict(states):
        qs = network(states)
        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = [qs.argmax().item()]
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            if qs.dim() == 1:
                x = (states.squeeze()[:, 0] == allowed_action_state).nonzero()
                if x.numel() == 0:
                    actions = [0]
                else:
                    actions = [x[qs[x].argmax().item()].item()]
            else:
                disallowed_actions_mask = (states[:, :, 0] != allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -1000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    # NETWORK TESTING
    results = []
    if return_history:
        history = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1
    results_master = []

    for j, test_graph in enumerate(graphs_test):

        i_comp = 0
        i_batch = 0
        t_total_rl_inference = 0

        n_spins = test_graph.shape[0]
        # Step Budget
        if optimisation_target == OptimisationTarget.MTDS:
            n_steps = int(n_spins * 2.5)
        else:
            n_steps = int(n_spins * step_factor)

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        test_env.immediate_reward_mode = "RL"

        # === 1. Greedy Strategy Setup ===
        print("Running greedy solver...", end="...")
        t_greedy_start = time.time()
        greedy_env = deepcopy(test_env)
        n = test_graph.shape[0]

        # Greedy 初始化状态
        if optimisation_target == OptimisationTarget.MTDS:
            greedy_env.reset(spins=np.ones(n))
        else:
            greedy_env.reset(spins=np.ones(n))

        # 制作一个干净的副本用于 Batch 复制
        test_test_env = deepcopy(greedy_env)

        greedy_env.immediate_reward_mode = "GREEDY"
        greedy_agent = Greedy(greedy_env)
        _, greedy_action_list = greedy_agent.solve(random_weight=greedy_env.random_weight)

        # Greedy +1 Init Raw Score (直接计数)
        raw_greedy_score = count_raw_size(greedy_env.best_spins, greedy_env.spin_basis)

        greedy_single_cut = raw_greedy_score
        greedy_single_spins = greedy_env.best_spins
        t_greedy_setup_time = time.time() - t_greedy_start
        print("done.")

        # Storage for RL
        best_cuts = []
        best_spins = []

        # Storage for Greedy Batch
        greedy_cuts = []
        greedy_spins = []
        greedy_times = []

        while i_comp < n_attempts:
            if max_batch_size is None:
                batch_size = 1
            else:
                batch_size = min(n_attempts - i_comp, max_batch_size)

            i_comp_batch = 0

            test_envs = [None] * batch_size
            if is_minimization:
                best_cuts_batch = [np.inf] * batch_size
            else:
                best_cuts_batch = [-np.inf] * batch_size

            best_spins_batch = [None] * batch_size
            greedy_envs = [None] * batch_size

            obs_batch = [None] * batch_size

            # 【修复点 1】初始化 Batch 临时列表
            greedy_cuts_batch = []
            greedy_spins_batch = []

            print(f"Preparing batch of {batch_size} environments...", end="...")

            for i in range(batch_size):
                env = deepcopy(test_test_env)
                # 强制显式 reset，保证 RL 和 Greedy 在同一起跑线
                obs_batch[i] = env.reset(reset_weight=False)
                test_envs[i] = env
                greedy_envs[i] = deepcopy(env)  # 复制给 Greedy 用
                best_spins_batch[i] = env.state[0, :env.n_spins].copy()

            print("done.")

            # --- RL Inference Loop ---
            t_start_rl = time.time()
            while i_comp_batch < batch_size:
                obs_batch_tensor = torch.FloatTensor(np.array(obs_batch)).to(device)
                actions = predict(obs_batch_tensor)
                obs_batch = []

                i = 0
                for env, action in zip(test_envs, actions):
                    if env is not None:
                        obs, rew, done, info = env.step(action, env.random_weight, adj_mask=False)

                        if not done:
                            obs_batch.append(obs)
                        else:
                            # RL Raw Score: 直接统计 best_spins 里的个数
                            if optimisation_target == OptimisationTarget.MTDS:
                                current_score = count_raw_size(env.best_spins, env.spin_basis)
                            else:
                                current_score = get_best_score(env, optimisation_target)

                            if is_minimization:
                                if current_score < best_cuts_batch[i]:
                                    best_cuts_batch[i] = current_score
                                    best_spins_batch[i] = env.best_spins.copy()
                            else:
                                if current_score > best_cuts_batch[i]:
                                    best_cuts_batch[i] = current_score
                                    best_spins_batch[i] = env.best_spins.copy()

                            i_comp_batch += 1
                            i_comp += 1
                            test_envs[i] = None
                    i += 1

            t_total_rl_inference += (time.time() - t_start_rl)
            i_batch += 1
            print("Finished agent testing batch {}.".format(i_batch))

            # --- Greedy Random Initialization Loop ---
            if env_args["reversible_spins"]:
                t_greedy_batch_start = time.time()
                print("Running greedy solver batch...", end="...")

                for env in greedy_envs:
                    # Greedy 也强制 Reset，确保状态一致
                    env.reset(reset_weight=False)
                    Greedy(env).solve(random_weight=None)

                    # Greedy Raw 直接计数，绝无 Remedy 干扰
                    if optimisation_target == OptimisationTarget.MTDS:
                        raw_cut = count_raw_size(env.best_spins, env.spin_basis)
                    else:
                        raw_cut = get_best_score(env, optimisation_target)

                    # 【修复点 2】添加到 Batch 列表，而不是主列表
                    greedy_cuts_batch.append(raw_cut)
                    greedy_spins_batch.append(env.best_spins.copy())

                t_greedy_batch = time.time() - t_greedy_batch_start
                greedy_times.append(t_greedy_batch)
                print("done.")

            best_cuts += best_cuts_batch
            best_spins += best_spins_batch

            # 【修复点 3】将 Batch 结果汇总到主列表
            if env_args["reversible_spins"]:
                greedy_cuts += greedy_cuts_batch
                greedy_spins += greedy_spins_batch

        # ============ Post-Processing & Selection ============

        # 1. Select Best RL Candidate (Based on Raw Score)
        if is_minimization:
            i_best = np.argmin(best_cuts)
        else:
            i_best = np.argmax(best_cuts)

        sol_rl = best_spins[i_best]
        # 使用直接计数的值作为 Raw
        if optimisation_target == OptimisationTarget.MTDS:
            best_cut_raw_true = count_raw_size(sol_rl, test_env.spin_basis)
        else:
            best_cut_raw_true = best_cuts[i_best]

        # 初始化 RL 变量
        score_rl_raw = best_cut_raw_true
        score_rl_remedy = score_rl_raw
        score_rl_pruned = score_rl_raw
        is_rl_raw_valid = False
        is_rl_final_valid = False
        t_rl_remedy = 0.0
        t_rl_prune = 0.0

        if optimisation_target == OptimisationTarget.MTDS:
            nx_graph = nx.from_numpy_array(test_graph)
            adj_list = nx.to_dict_of_lists(nx_graph)

            # (1) Check Raw Validity (使用 target_val_def)
            selected_nodes_raw_rl = {i for i, x in enumerate(sol_rl) if x == target_val_def}
            is_rl_raw_valid = is_valid_mtds_check(nx_graph, selected_nodes_raw_rl)

            # (2) Apply Remedy
            t_rem_start = time.time()
            remedied_sol_rl, _ = test_env.apply_mtds_remedy(sol_rl)
            t_rl_remedy = time.time() - t_rem_start
            score_rl_remedy = calculate_mtds_size(test_env.matrix, remedied_sol_rl, test_env.spin_basis)

            # (3) Apply Pruning
            t_prune_start = time.time()
            # 直接使用定义的 target_val_def
            selected_nodes_rem_rl = {i for i, x in enumerate(remedied_sol_rl) if x == target_val_def}
            pruned_nodes_rl = prune_solution_fast(adj_list, selected_nodes_rem_rl)

            score_rl_pruned = len(pruned_nodes_rl)
            is_rl_final_valid = is_valid_mtds_check(nx_graph, pruned_nodes_rl)
            t_rl_prune = time.time() - t_prune_start
        else:
            is_rl_raw_valid = True
            is_rl_final_valid = True

        # ========================================================
        # 2. Select Best Greedy Candidate & Process
        # ========================================================
        greedy_best_sol = None
        greedy_best_raw_initial = 0
        avg_greedy_time = 0.0

        if env_args["reversible_spins"] and len(greedy_cuts) > 0:
            i_greedy = np.argmin(greedy_cuts) if is_minimization else np.argmax(greedy_cuts)
            greedy_best_raw_initial = greedy_cuts[i_greedy]
            greedy_best_sol = greedy_spins[i_greedy]
            if n_attempts > 0:
                avg_greedy_time = np.sum(greedy_times) / n_attempts
        else:
            greedy_best_raw_initial = greedy_single_cut
            greedy_best_sol = greedy_single_spins
            avg_greedy_time = t_greedy_setup_time

        # 再次确保 Greedy Raw 是计算出来的真实值
        if optimisation_target == OptimisationTarget.MTDS and greedy_best_sol is not None:
            score_greedy_raw = count_raw_size(greedy_best_sol, test_env.spin_basis)
        else:
            score_greedy_raw = greedy_best_raw_initial

        # 初始化 Greedy 变量
        score_greedy_remedy = score_greedy_raw
        score_greedy_pruned = score_greedy_raw
        is_gr_raw_valid = False
        is_gr_final_valid = False
        t_gr_remedy = 0.0
        t_gr_prune = 0.0

        if optimisation_target == OptimisationTarget.MTDS and greedy_best_sol is not None:
            if 'nx_graph' not in locals():
                nx_graph = nx.from_numpy_array(test_graph)
                adj_list = nx.to_dict_of_lists(nx_graph)

            # (1) Check Greedy Raw Validity
            selected_nodes_raw_gr = {i for i, x in enumerate(greedy_best_sol) if x == target_val_def}
            is_gr_raw_valid = is_valid_mtds_check(nx_graph, selected_nodes_raw_gr)

            # (2) Apply Remedy
            t_g_rem_start = time.time()
            remedied_sol_gr, _ = test_env.apply_mtds_remedy(greedy_best_sol)
            t_gr_remedy = time.time() - t_g_rem_start
            score_greedy_remedy = calculate_mtds_size(test_env.matrix, remedied_sol_gr, test_env.spin_basis)

            # (3) Apply Pruning
            t_g_prune_start = time.time()
            selected_nodes_rem_gr = {i for i, x in enumerate(remedied_sol_gr) if x == target_val_def}
            pruned_nodes_gr = prune_solution_fast(adj_list, selected_nodes_rem_gr)

            score_greedy_pruned = len(pruned_nodes_gr)
            is_gr_final_valid = is_valid_mtds_check(nx_graph, pruned_nodes_gr)
            t_gr_prune = time.time() - t_g_prune_start
        else:
            is_gr_raw_valid = True
            is_gr_final_valid = True

        # Times Calculation
        avg_rl_inference = t_total_rl_inference / n_attempts if n_attempts > 0 else 0
        time_rl_total = avg_rl_inference + t_rl_remedy
        time_greedy_total = avg_greedy_time + t_gr_remedy

        # ==============================================================================
        # [新增] 提取最终解的节点编号 (RL + Remedy + Pruned)
        # ==============================================================================
        # pruned_nodes_rl 是一个包含节点索引的集合 (Set)，例如 {0, 5, 8}
        # 我们将其转为排序后的列表，再转为字符串保存，例如 "[0, 5, 8]"
        rl_nodes_str = "[]"
        if 'pruned_nodes_rl' in locals() and pruned_nodes_rl is not None:
            rl_nodes_str = str(sorted(list(pruned_nodes_rl)))

        greedy_nodes_str = "[]"
        if 'pruned_nodes_gr' in locals() and pruned_nodes_gr is not None:
            greedy_nodes_str = str(sorted(list(pruned_nodes_gr)))
        # ==============================================================================

        # Formatting Output
        rl_raw_str = f"{score_rl_raw}({'V' if is_rl_raw_valid else 'X'})"
        gr_raw_str = f"{score_greedy_raw}({'V' if is_gr_raw_valid else 'X'})"

        print(
            'Graph {}, SOL: {} | RL: Raw={}->Rem={} | Gr: Raw={}->Rem={} || '
            'Time(Avg): RL={:.4f}, Gr={:.4f}'.format(
                j, "MTDS",
                rl_raw_str, score_rl_remedy,
                gr_raw_str, score_greedy_remedy,
                time_rl_total, time_greedy_total))

        results_master.append({
            'graph_id': j,
            'target': 'MTDS',
            'score_rl_raw': score_rl_raw,
            'valid_rl_raw': is_rl_raw_valid,
            'score_rl_remedy': score_rl_remedy,
            'score_rl_pruned': score_rl_pruned,
            'valid_rl_final': is_rl_final_valid,
            'rl_solution_nodes': rl_nodes_str,
            'score_greedy_raw': score_greedy_raw,
            'valid_greedy_raw': is_gr_raw_valid,
            'score_greedy_remedy': score_greedy_remedy,
            'score_greedy_pruned': score_greedy_pruned,
            'valid_greedy_final': is_gr_final_valid,
            'greedy_solution_nodes': greedy_nodes_str,
            'time_rl_total': np.round(time_rl_total, 4),
            'time_greedy_total': np.round(time_greedy_total, 4),
        })

    results_master = pd.DataFrame(results_master)

    print("\n" + "=" * 160)
    print("[DEBUG] Batched Test Results Summary")
    print("=" * 160)
    cols = ['graph_id',
            'score_rl_raw', 'valid_rl_raw', 'score_rl_remedy','rl_solution_nodes',
            'score_greedy_raw', 'valid_greedy_raw', 'score_greedy_remedy',
            'time_rl_total', 'time_greedy_total']
    print(results_master[cols].to_string(index=False))
    print("=" * 160 + "\n")

    if return_raw == False and return_history == False:
        return results
    else:
        return results_master


def __test_network_sequential(network, env_args, graphs_test, step_factor=1,
                              n_attempts=50, return_raw=False, return_history=False):
    if return_raw or return_history:
        raise NotImplementedError("Sequential does not support history yet.")

    optimisation_target = env_args.get('optimisation_target', OptimisationTarget.DSP)
    is_minimization = should_minimize(optimisation_target)
    results = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for i, test_graph in enumerate(graphs_test):
        n_steps = int(test_graph.shape[0] * step_factor)

        # Initialize boundaries
        best_cut = np.inf if is_minimization else -np.inf
        best_spins = []

        greedy_random_cut = np.inf if is_minimization else -np.inf
        greedy_random_spins = []

        times = []

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        net_agent = Network(network, test_env,
                            record_cut=False, record_rewards=False, record_qs=False)

        # Greedy Init (No Remedy)
        greedy_env = deepcopy(test_env)
        n = test_graph.shape[0]
        if optimisation_target == OptimisationTarget.MTDS:
            # MTDS Random Initialization
            if env_args['spin_basis'] == SpinBasis.BINARY:
                rand = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
            else:
                rand = np.random.choice([-1, 1], size=n, p=[0.3, 0.7])
            greedy_env.reset(spins=rand)
        else:
            greedy_env.reset(spins=np.array([1] * n))

        greedy_agent = Greedy(greedy_env)
        greedy_agent.solve(random_weight=greedy_env.random_weight)

        raw_greedy = get_best_score(greedy_env, optimisation_target)
        if optimisation_target == OptimisationTarget.MTDS and raw_greedy <= 0:
            greedy_single_cut = np.inf
        else:
            greedy_single_cut = raw_greedy
        greedy_single_spins = greedy_env.best_spins

        for k in range(n_attempts):
            net_agent.reset(clear_history=True, reset_weight=False)
            greedy_env = deepcopy(test_env)
            greedy_agent = Greedy(greedy_env)

            tstart = time.time()
            net_agent.solve(random_weight=greedy_env.random_weight)
            times.append(time.time() - tstart)

            # Get RL Raw Score (No Remedy)
            cut = get_best_score(test_env, optimisation_target)

            # Handle Invalid Solutions
            is_valid = True
            if optimisation_target == OptimisationTarget.MTDS and cut <= 0:
                is_valid = False
                cut = np.inf

            # Comparison Logic
            if is_minimization:
                if is_valid and cut < best_cut:
                    best_cut = cut
                    best_spins = test_env.best_spins.copy()
            else:
                if cut > best_cut:
                    best_cut = cut
                    best_spins = test_env.best_spins.copy()

            # Greedy Random Solve
            greedy_agent.solve()
            greedy_cut = get_best_score(greedy_env, optimisation_target)

            if optimisation_target == OptimisationTarget.MTDS and greedy_cut <= 0:
                greedy_cut = np.inf

            if is_minimization:
                if greedy_cut < greedy_random_cut:
                    greedy_random_cut = greedy_cut
            else:
                if greedy_cut > greedy_random_cut:
                    greedy_random_cut = greedy_cut

            print(
                '\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut: {} / {}\t\t\t'.format(
                    i + 1, k, n_attempts, best_cut, greedy_random_cut, greedy_single_cut),
                end=".")

        # ============ MTDS FINAL REMEDY (Post-Processing) ============
        if optimisation_target == OptimisationTarget.MTDS:
            # If best_spins is empty (never found valid solution), use last state
            if len(best_spins) == 0:
                best_spins = test_env.state[0, :test_env.n_spins].copy()

            remedied, _ = test_env.apply_mtds_remedy(best_spins)
            final_score = calculate_mtds_size(
                test_env.matrix, remedied, test_env.spin_basis
            )
            # This final_score is RL + Remedy
            rl_final_display = final_score
        else:
            rl_final_display = best_cut

        results.append([rl_final_display, best_spins,
                        greedy_single_cut, [],
                        greedy_random_cut, [],
                        np.mean(times)])

    return pd.DataFrame(data=results, columns=["cut", "sol",
                                               "greedy (+1 init) cut", "greedy (+1 init) sol",
                                               "greedy (rand init) cut", "greedy (rand init) sol",
                                               "time"])


####################################################
# LOADING GRAPHS
####################################################

Graph = namedtuple('Graph', 'name n_vertices n_edges matrix bk_val bk_sol')


def load_graph(graph_dir, graph_name):
    inst_loc = os.path.join(graph_dir, 'instances', graph_name + '.mc')
    val_loc = os.path.join(graph_dir, 'bkvl', graph_name + '.bkvl')
    sol_loc = os.path.join(graph_dir, 'bksol', graph_name + '.bksol')

    vertices, edges, matrix = 0, 0, None
    bk_val, bk_sol = None, None

    with open(inst_loc) as f:
        for line in f:
            arr = list(map(int, line.strip().split(' ')))
            if len(arr) == 2:  # contains the number of vertices and edges
                n_vertices, n_edges = arr
                matrix = np.zeros((n_vertices, n_vertices))
            else:
                assert type(matrix) == np.ndarray, 'First line in file should define graph dimensions.'
                i, j, w = arr[0] - 1, arr[1] - 1, arr[2]
                matrix[[i, j], [j, i]] = w

    with open(val_loc) as f:
        bk_val = float(f.readline())

    with open(sol_loc) as f:
        bk_sol_str = f.readline().strip()
        bk_sol = np.array([int(x) for x in list(bk_sol_str)] + [np.random.choice([0, 1])])  # final spin is 'no-action'

    return Graph(graph_name, n_vertices, n_edges, matrix, bk_val, bk_sol)


def load_graph_set(graph_save_loc):
    graphs_test = pickle.load(open(graph_save_loc, 'rb'))

    def graph_to_array(g):
        if type(g) == nx.Graph:
            g = nx.to_numpy_array(g)
        elif type(g) == sp.sparse.csr_matrix:
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_loc))
    return graphs_test


####################################################
# FILE UTILS
####################################################

def mk_dir(export_dir, quite=False):
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
            print('created dir: ', export_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        print('dir already exists: ', export_dir)

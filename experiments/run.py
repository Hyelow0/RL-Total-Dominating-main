import argparse
import os
import sys
import pickle

from numba.core.cgutils import false_bit

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（RL-Dominating-main）
project_root = os.path.dirname(current_dir)
# 添加到 Python 路径
sys.path.insert(0, project_root)
from src.envs.utils import OptimisationTarget
from train import run as train_run
from test import run as test_run
import optuna


class DSPModel:
    def __init__(self, n_spins, graph_type, args: argparse.Namespace, optimisation_target=OptimisationTarget.MTDS, save_loc='../checkpoints/', load_loc='../data/'):
        self.n_spins = n_spins
        self.type = graph_type
        self.original_save_loc = save_loc
        self.save_loc = save_loc
        self.original_load_loc = load_loc
        self.args = args
        self.optimisation_target = optimisation_target
        if n_spins <= 0:
            raise Exception('Parameter n_spins must larger than 0!')
        if graph_type not in ['ER', 'ER_SOL', 'BA', 'grid', 'tri', 'hex', 'random', 'large', 'whole', 'NI']:
            raise Exception(
                "No type called {}, graph type must be 'ER', 'ER_SOL', 'BA', 'GRID', 'TRI', 'HEX', 'large', 'whole', 'NI".format(graph_type))
        if type(args) != argparse.Namespace:
            raise TypeError("Parameter 'args' must be type 'argparse.Namespace'!")
        self.set_paths()

    def set_paths(self):
        self.save_loc = self.original_save_loc + self.type + '_graphs/' + str(self.n_spins) + 'spins/'
        self.load_loc = self.original_load_loc + self.type + '_graphs/'

    def train(self, timesteps, verbose=True):
        target_name = "MTDS" if self.optimisation_target == OptimisationTarget.MTDS else "DSP"
        print(f"[DEBUG] Training {target_name}: n_spins={self.n_spins}, type={self.type}")
        print(f"[DEBUG] Save location: {self.save_loc}")
        print(f"[DEBUG] Load location: {self.load_loc}")
        score = train_run(timesteps=timesteps, n_spins=self.n_spins, test_loc=self.load_loc, save_loc=self.save_loc,
                           args=self.args, optimisation_target=self.optimisation_target, verbose=verbose, graph_type=self.type)
        return score

    def test(self, test_model=None, custom_model_path=None):
        if test_model not in [None, '20', '100']:
            pass
        if self.type == 'NI':
            spins = [20, 40, 80, 100, 200, 300, 400, 500]
            # 构造测试集路径
            # 注意：请确保你的 ../data/NI_graphs/20spins/ 目录下有 random_graphs.pkl 文件
            # 如果没有，test.py 会报错。你需要先运行 generate_data.py (如果有的话)
            graph_path = [self.load_loc + str(spins[j]) + 'spins/random_graphs.pkl' for j in range(len(spins))]

            for spin, path in zip(spins, graph_path):
                # 如果文件不存在，跳过
                if not os.path.exists(path):
                    if spin == self.n_spins:  # 当前训练的节点数必须要有测试集
                        print(f"[WARNING] Test file not found for {spin} spins: {path}")
                    continue

                max_batch_size = None
                if self.n_spins >= 400:
                    max_batch_size = 5

                # 构造保存结果的路径，复用当前 save_loc 但替换 spin 数
                current_save_loc = self.save_loc.replace(str(self.n_spins), str(spin))

                print(f"\n[TEST] Testing on NI graphs: {spin} spins...")
                test_run(save_loc=current_save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                         args=self.args, optimisation_target=self.optimisation_target, model_name=test_model,
                         custom_model_path=custom_model_path)
        if self.type == 'ER':
            p = ['p0.1', 'p0.3', 'p0.5', 'p0.8']
            spins = [20, 40, 80, 100, 200, 300, 400, 500]
            for i in range(len(p)):
                graph_path = [self.load_loc + str(spins[j]) + 'spins/ER_' + p[i] + '.pkl' for j in range(len(spins))]
                for spin, path in zip(spins, graph_path):
                    max_batch_size = None
                    if spin >= 400:
                        max_batch_size = 5
                    save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                    test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                             args=self.args, optimisation_target=self.optimisation_target, p=p[i], model_name=test_model, custom_model_path=custom_model_path)
        elif self.type == 'BA':
            m = ['m4', 'm8', 'm12', 'm18']
            spins = [20, 40, 80, 100, 200, 300, 400, 500]
            for i in range(len(m)):
                graph_path = [self.load_loc + str(spins[j]) + 'spins/BA_' + m[i] + '.pkl' for j in range(len(spins))]
                for spin, path in zip(spins, graph_path):
                    max_batch_size = None
                    if spin >= 400:
                        max_batch_size = 5
                    save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                    test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                             args=self.args, optimisation_target=self.optimisation_target, m=m[i], model_name=test_model, custom_model_path=custom_model_path)

        elif self.type == 'random':
            spins = [20, 40, 80, 100, 200, 300, 400, 500]
            graph_path = [self.load_loc + str(spins[j]) + 'spins/random_graphs.pkl' for j in range(len(spins))]
            for spin, path in zip(spins, graph_path):
                max_batch_size = None
                if self.n_spins >= 400:
                    max_batch_size = 5
                save_loc = self.save_loc.replace(str(self.n_spins), str(spin))
                test_run(save_loc=save_loc, graph_save_loc=path, batched=True, max_batch_size=max_batch_size,
                         args=self.args, optimisation_target=self.optimisation_target, model_name=test_model, custom_model_path=custom_model_path)

        elif self.type == 'ER_SOL':
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/ER_' + file + '.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args, optimisation_target=self.optimisation_target, model_name=test_model, custom_model_path=custom_model_path)

        elif self.type == 'large':
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/graph.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args, optimisation_target=self.optimisation_target,
                         model_name=test_model, custom_model_path=custom_model_path)

        elif self.type in ['grid', 'tri', 'hex']:
            file_name = os.listdir(self.load_loc)
            for file in file_name:
                graph_path = self.load_loc + file + '/graph.pkl'
                save_loc = self.original_save_loc + self.type + '_graphs/' + file + '/'
                test_run(save_loc=save_loc, graph_save_loc=graph_path, batched=True, args=self.args, optimisation_target=self.optimisation_target,
                         model_name=test_model, custom_model_path=custom_model_path)

    def BA_ER_train_test(self, train_steps):
        if self.type == 'grid':
            graph_type = ['BA', 'ER']
            for ttype in graph_type:
                self.type = ttype
                self.set_paths()
                self.train(timesteps=train_steps)
                self.test()
        else:
            raise TypeError(
                "function BA_ER_train_test must used for graph type 'whole' , but got type '" + self.type + "'!")

    def para_optim_train(self, trail):
        gamma = trail.suggest_float('gamma', 0.95, 1)
        init_weight_std = trail.suggest_float('init_weight_std', 1e-5, 1e-3)
        replay_buffer_size = trail.suggest_int('replay_buffer_size', 50000, 500000)
        update_target_frequency = trail.suggest_int('update_target_frequency', 5000, 30000)
        lr = trail.suggest_float('lr', 1e-6, 1)
        minibatch_size = trail.suggest_int('minibatch_size', 16, 128)
        final_exploration_rate = trail.suggest_float('final_exploration_rate', 0, 0.2)
        parser.set_defaults(gamma=gamma)
        parser.set_defaults(init_weight_std=init_weight_std)
        parser.set_defaults(replay_buffer_size=replay_buffer_size)
        parser.set_defaults(update_target_frequency=update_target_frequency)
        parser.set_defaults(lr=lr)
        parser.set_defaults(minibatch_size=minibatch_size)
        parser.set_defaults(final_exploration_rate=final_exploration_rate)
        self.args = parser.parse_args()
        scores = self.train(timesteps=50000, verbose=False)
        return scores

    def optim(self):
        storage_name = 'sqlite:///optuna.db'
        study = optuna.create_study(direction="maximize",
                                    study_name="DSP_OPTIM", storage=storage_name, load_if_exists=True
                                    )
        study.optimize(self.para_optim_train, n_trials=200)
        best_para = study.best_params
        fw = open(self.save_loc + 'best_model_para.pkl', 'wb')
        pickle.dump(best_para, fw)
        fw.close()
        parser.set_defaults(gamma=best_para['gamma'])
        parser.set_defaults(init_weight_std=best_para['init_weight_std'])
        parser.set_defaults(replay_buffer_size=best_para['replay_buffer_size'])
        parser.set_defaults(update_target_frequency=best_para['update_target_frequency'])
        parser.set_defaults(lr=best_para['lr'])
        parser.set_defaults(minibatch_size=best_para['minibatch_size'])
        parser.set_defaults(final_exploration_rate=best_para['final_exploration_rate'])
        self.args = parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSP/MTDS Args')
    parser.add_argument('--gamma', type=float, default=0.99, help='Decrease rate of reward.')
    parser.add_argument('--norm_reward', default=False, help='Reward of reinforcement learning normalization or not.')
    parser.add_argument('--stag_punishment', default=1.0,
                        help='Punishment of steps of reinforcement learning or not.')
    parser.add_argument('--if_weight', default=False, help='Weighted graph experiments.')
    parser.add_argument('--init_weight_std', type=float, default=0.01, help='Init weight std of linear.')
    parser.add_argument('--replay_buffer_size', type=int, default=200000, help='Size of replay buffer.')
    parser.add_argument('--update_target_frequency', type=int, default=10000, help='Frequency of update target network.')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr.')#1e-4 #5e-5
    parser.add_argument('--minibatch_size', type=int, default=32, help='Mini batch size')#32
    parser.add_argument('--initial_exploration_rate', type=float, default=1.0, help='Initial exploration rate.')#1.0
    parser.add_argument('--final_exploration_rate', type=float, default=0.05, help='Final exploration rate.')#0.05
    parser.add_argument('--final_exploration_step', type=float, default=600000, help='Final exploration step.')#800000#400000#300000
    parser.add_argument('--grid', default=False, help='Grid tri or hex grid graphs experiments.')
    parser.add_argument('--mtds_constraint_penalty', type=float, default=3.0, help = 'MTDS constraint violation penalty coefficient.')
    parser.add_argument('--test_episodes', type=int, default=1, help='Number of test episodes per graph.')
    args = parser.parse_args()
    args.test_episodes = 5
    print(f"[INFO] Force set test_episodes = {args.test_episodes} for deterministic evaluation.")
    # model = DSPModel(n_spins=20, graph_type='NI', args=args, optimisation_target=OptimisationTarget.MTDS)

    n_train_spins = 20
    model = DSPModel(n_spins=n_train_spins, graph_type='NI', args=args, optimisation_target=OptimisationTarget.MTDS)
    # model.optim()
    # print("\n>>> START TRAINING (Reversible Spins + MTDS) >>>")
    # model.train(timesteps=800000)

    exist_mode = True

    print("\n>>> START TRANSFER TESTING on Grid Graphs >>>")
    try:

        if exist_mode:
            ba_model_path = os.path.join(project_root, 'checkpoints', 'record_stage1_new', f'20spins',
                                         f'network_best_20_stage1_new.pth')
        else:
            ba_model_path = os.path.join(project_root, 'checkpoints', 'NI_graphs', f'{n_train_spins}spins',
                                         f'network_best_{n_train_spins}.pth')
        if os.path.exists(ba_model_path):
            model.type = 'grid'
            model.set_paths()
            model.test(test_model=str(n_train_spins), custom_model_path=ba_model_path)
        else:
            print(f"Model file not found for transfer learning: {ba_model_path}")
    except Exception as e:
        print(f"Skipping Grid test: {e}")

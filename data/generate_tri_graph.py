import networkx as nx
import pickle
import os


class TriGraphGenerator:
    def __init__(self, target_nodes, set_size, base_dir):
        self.target_nodes = target_nodes
        self.set_size = set_size
        self.base_dir = base_dir

    def calculate_real_nodes(self, m, n):
        row_count = m + 1
        N = (n + 1) // 2
        col_count = N + 1
        total = row_count * col_count

        if n % 2 != 0:
            odd_rows_count = (m + 1) // 2
            total -= odd_rows_count

        return total

    def find_exact_params(self):
        best_m, best_n = None, None
        min_diff_ratio = float('inf')

        for m in range(2, 100):
            for n in range(2, 100):
                calc_nodes = self.calculate_real_nodes(m, n)

                if calc_nodes == self.target_nodes:
                    ratio = max(m, n) / min(m, n)
                    if ratio < min_diff_ratio:
                        min_diff_ratio = ratio
                        best_m, best_n = m, n

        return best_m, best_n

    def generate(self):

        m, n = self.find_exact_params()

        if m is None:
            return


        graphs = []
        for _ in range(self.set_size):
            g = nx.triangular_lattice_graph(m, n)

            real_n = g.number_of_nodes()
            if real_n != self.target_nodes:
                print(f"❌ 尺寸不匹配 (预期{self.target_nodes}, 实际{real_n})，跳过。")
                return

            graphs.append(g)
        folder_name = f"{self.target_nodes}spins"
        save_path = os.path.join(self.base_dir, folder_name, 'graph.pkl')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(graphs, f)


if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'tri_graphs')

    mandatory_nodes = [20, 30, 40, 50, 60, 80, 100, 120, 160, 200]

    final_targets = sorted(list(set(mandatory_nodes)))

    SET_SIZE = 1


    for target in final_targets:
        generator = TriGraphGenerator(target, SET_SIZE, output_dir)
        generator.generate()
import pickle
import random
import pandas as pd
import networkx as nx
import numpy as np
import os


class SetGraphGrenerator:
    def __init__(self, n_spins, style, size, file_path, density=None, m=None, p=None, grid_n=None, grid_m=None):
        self.n_spins = n_spins
        self.density = density
        self.path = file_path
        self.set_size = size
        self.style = style
        self.m = m
        self.p = p
        self.grid_n = grid_n
        self.grid_m = grid_m

    def get_matrix(self):
        matrix = np.zeros((self.n_spins, self.n_spins))
        density = np.random.uniform()
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    w = random.choice([0, 1])
                    matrix[i, j] = w
                    matrix[j, i] = w
        return matrix

    def get_NI_graph(self):
        p = self.p if self.p is not None else 0.15
        while True:
            g = nx.erdos_renyi_graph(self.n_spins, p)
            degrees = [d for n, d in g.degree()]
            if 0 not in degrees:
                break
        return nx.to_numpy_array(g)

    def get_GRID_graph(self):
        if self.grid_n is not None and self.grid_m is not None:
            return nx.grid_graph((self.grid_n, self.grid_m))
        else:
            raise Exception('generate grid graph need parameter n and m!')

    def get_BA_graph(self):
        return nx.barabasi_albert_graph(self.n_spins, self.m)

    def get_ER_graph(self):
        g = nx.erdos_renyi_graph(self.n_spins, self.p)
        while not nx.is_connected(g):
            g = nx.erdos_renyi_graph(self.n_spins, self.p)
        return g

    def get_TRIGRID_graph(self):
        g = nx.triangular_lattice_graph(self.grid_n, self.grid_m)
        real_n = g.number_of_nodes()
        if self.n_spins != real_n:
            self.n_spins = real_n
        return g

    def get_HEXGRID_graph(self):
        g = nx.hexagonal_lattice_graph(self.grid_n, self.grid_m)
        real_n = g.number_of_nodes()
        if self.n_spins != real_n:
            self.n_spins = real_n
        return g

    # ====================================================================

    def get_graph_set(self):
        if self.style == 'random':
            return [self.get_matrix() for _ in range(self.set_size)]
        elif self.style == 'NI':
            return [self.get_NI_graph() for _ in range(self.set_size)]
        elif self.style == 'BA':
            return [self.get_BA_graph() for _ in range(self.set_size)]
        elif self.style == 'ER':
            return [self.get_ER_graph() for _ in range(self.set_size)]
        elif self.style == 'Grid':
            return [self.get_GRID_graph() for _ in range(self.set_size)]
        elif self.style == 'TriGrid':
            return [self.get_TRIGRID_graph() for _ in range(self.set_size)]
        elif self.style == 'HexGrid':
            return [self.get_HEXGRID_graph() for _ in range(self.set_size)]
        else:
            raise Exception('No style called ' + self.style)

    def generate_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        graph_set = self.get_graph_set()

        if len(graph_set) > 0:
            sample = graph_set[0]
            if isinstance(sample, nx.Graph):
                print(
                    f"[Check] Saved to .../{os.path.basename(os.path.dirname(self.path))}/ | Real Nodes: {sample.number_of_nodes()}")
            else:
                print(
                    f"[Check] Saved to .../{os.path.basename(os.path.dirname(self.path))}/ | Real Nodes: {sample.shape[0]}")

        with open(self.path, 'wb') as f:
            pickle.dump(graph_set, f)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    set_size = 1

    grid_configs = [
        (4, 5, 20),
        (6, 6, 36),
        (7, 9, 63),
        (8, 9, 72),
        (9, 9, 81),
        (10, 10, 100),
        (10, 12, 120),
        (8, 16, 128),
    ]
    for grid_n, grid_m, n_spins in grid_configs:
        file_path = os.path.join(data_dir, 'grid_graphs', f'{n_spins}spins', 'graph.pkl')
        generator = SetGraphGrenerator(n_spins, 'Grid', set_size, file_path, grid_n=grid_n, grid_m=grid_m)
        generator.generate_file()

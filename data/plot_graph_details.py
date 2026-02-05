import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ast
import os
import pickle
import argparse

# ==========================================
# 1. 参数设置
# ==========================================
parser = argparse.ArgumentParser(description="Visualize RL Solution (Bottom Aligned)")
parser.add_argument('--excel', type=str, required=True, help='Path to the results Excel file')
parser.add_argument('--data_root', type=str, default='../data', help='Root directory of graph data')
parser.add_argument('--idx', type=int, default=0, help='Row index in Excel (Graph ID)')
args = parser.parse_args()

# --- 样式常量 ---
NODE_SIZE = 250  # 节点大小
EDGE_WIDTH = 2.0  # 边框粗细
NODE_BORDER_WIDTH = 1.5
FONT_SIZE = 20

# 【参数 1】固定画布尺寸 (英寸)
FIXED_FIG_W = 5.5
FIXED_FIG_H = 4.5

# 【参数 2】固定视野范围 (Data Units)
VIEW_WIDTH_SPAN = 7.0
VIEW_HEIGHT_SPAN = 7.0

# 【参数 3】底部留白 (Data Units)
# 这是一个关键参数。它决定了图的最低点 (y_min) 距离画布下边缘的距离。
# 我们必须留出足够的空间来放标签。
# 设为 2.0，意味着 y_min 下方有 2.0 单位的空白区域。
SPACE_BELOW_GRAPH = 1.5

# 【参数 4】文字偏移量
# 文字位于 y_min 下方多少单位
TEXT_OFFSET = 0.8


# ==========================================
# 2. 辅助函数
# ==========================================
def extract_info_from_path(path):
    parts = path.replace('\\', '/').split('/')
    n_spins = None
    graph_type = None
    for part in parts:
        if 'spins' in part and part[:-5].isdigit():
            n_spins = int(part[:-5])
        if '_graphs' in part:
            graph_type = part
    return graph_type, n_spins


def load_original_graph(data_root, graph_type, n_spins, graph_id):
    if graph_type in ['grid_graphs', 'tri_graphs', 'hex_graphs']:
        filename = 'graph.pkl'
    else:
        filename = 'random_graphs.pkl'
    pkl_path = os.path.join(data_root, graph_type, f'{n_spins}spins', filename)

    if not os.path.exists(pkl_path):
        dir_path = os.path.dirname(pkl_path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
            if len(files) > 0:
                pkl_path = os.path.join(dir_path, files[0])
            else:
                raise FileNotFoundError(f"No .pkl found in {dir_path}")
        else:
            raise FileNotFoundError(f"Path not found: {dir_path}")

    print(f"[Info] Loading graph from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs[graph_id]


def get_triangular_pos(n_spins):
    import math
    cols = int(math.sqrt(n_spins))
    while n_spins % cols != 0: cols -= 1
    if cols == 0: cols = 1
    if cols == 1 and n_spins > 5: cols = int(np.ceil(np.sqrt(n_spins)))

    pos = {}
    for i in range(n_spins):
        r = i // cols
        c = i % cols
        x = c + 0.5 * (r % 2)
        y = -r * (np.sqrt(3) / 2)
        pos[i] = (x, y)
    return pos


def get_node_pos(G, graph_type, n_spins):
    if isinstance(G, nx.Graph):
        pos = nx.get_node_attributes(G, 'pos')
        if pos: return {i: pos[n] for i, n in enumerate(sorted(G.nodes()))}
    if 'tri' in graph_type: return get_triangular_pos(n_spins)
    import math
    cols = int(math.sqrt(n_spins))
    while n_spins % cols != 0: cols -= 1
    if cols == 1: cols = int(np.ceil(np.sqrt(n_spins)))
    pos = {}
    for i in range(n_spins):
        r = i // cols
        c = i % cols
        pos[i] = (c, -r)
    return pos


# ==========================================
# 3. 主逻辑
# ==========================================
graph_type, n_spins = extract_info_from_path(args.excel)
if not os.path.exists(args.excel): exit()

df = pd.read_excel(args.excel)
try:
    node_str = df.iloc[args.idx]['rl_solution_nodes']
    solution_nodes = ast.literal_eval(node_str)
except KeyError:
    exit()

raw_data = load_original_graph(args.data_root, graph_type, n_spins, args.idx)

if isinstance(raw_data, np.ndarray):
    G_vis = nx.from_numpy_array(raw_data)
else:
    mapping = {old: new for new, old in enumerate(sorted(raw_data.nodes()))}
    G_vis = nx.relabel_nodes(raw_data, mapping)

pos = get_node_pos(raw_data, graph_type, n_spins)

pos_vertical = {node: (coords[1], coords[0]) for node, coords in pos.items()}
xs = [coord[0] for coord in pos_vertical.values()]
ys = [coord[1] for coord in pos_vertical.values()]

node_colors = []
node_sizes = []
solution_set = set(solution_nodes)

for node in sorted(G_vis.nodes()):
    if node in solution_set:
        node_colors.append('red')
        node_sizes.append(NODE_SIZE)
    else:
        node_colors.append('white')
        node_sizes.append(NODE_SIZE)

    # ==========================================
# 4. 绘图 (底部对齐逻辑)
# ==========================================
fig, ax = plt.subplots(figsize=(FIXED_FIG_W, FIXED_FIG_H))

nx.draw_networkx_edges(G_vis, pos_vertical, edge_color='black', width=EDGE_WIDTH, alpha=1.0, ax=ax)
nx.draw_networkx_nodes(G_vis, pos_vertical, node_color=node_colors, node_size=node_sizes, edgecolors='black',
                       linewidths=NODE_BORDER_WIDTH, ax=ax)

# 获取当前图的几何边界
y_min_actual = min(ys)
center_x = (min(xs) + max(xs)) / 2

# 【关键修改】设置坐标轴范围 (View Box)
# 1. X轴：保持水平居中
ax.set_xlim(center_x - VIEW_WIDTH_SPAN / 2, center_x + VIEW_WIDTH_SPAN / 2)

# 2. Y轴：不再居中，而是锚定底部
# 让视图的下边界固定在 y_min_actual - SPACE_BELOW_GRAPH 的位置
# 这样，无论图多高，最低点 y_min 总是距离视图底部 SPACE_BELOW_GRAPH 那么远
y_bottom_view = y_min_actual - SPACE_BELOW_GRAPH
ax.set_ylim(y_bottom_view, y_bottom_view + VIEW_HEIGHT_SPAN)

ax.set_aspect('equal')
ax.axis('off')

# --- 底部文本 ---
if 'tri_graphs' in graph_type:
    formatted_type = "TRIANGULAR LATTICE GRAPH"
elif 'grid_graphs' in graph_type:
    formatted_type = "4×5 GRID GRAPH"
else:
    formatted_type = graph_type.replace('_graphs', '').replace('_', ' ').upper() + " GRAPH"

bottom_text = f"{n_spins} VERTICES\n{formatted_type}"

# 文字位置：相对于最低点下移 TEXT_OFFSET
# 因为视图底部是 y_min - SPACE_BELOW_GRAPH，而文字在 y_min - TEXT_OFFSET
# 所以文字距离视图底部的距离是 (SPACE_BELOW_GRAPH - TEXT_OFFSET)，这是个恒定值
ax.text(
    center_x,
    y_min_actual - TEXT_OFFSET,
    bottom_text,
    ha='center',
    va='top',
    fontsize=FONT_SIZE,
    fontweight='black',
    fontstretch='ultra-condensed',
    linespacing=1.2,
    clip_on=False
)

save_dir = os.path.dirname(args.excel)
# save_name = os.path.join(save_dir, f"vis_{graph_type}_N{n_spins}_bottom_aligned.pdf")
#
# # 必须保留原始画布
# plt.savefig(save_name, dpi=300)
# print(f"[Done] Saved to: {save_name}")

# 【修改处】 将后缀改为 .png
save_name = os.path.join(save_dir, f"vis_{graph_type}_N{n_spins}_bottom_aligned.png")

# 【修改处】 format='png'，dpi=300 保证清晰度
plt.savefig(save_name, dpi=300, format='png')
print(f"[Done] Saved to: {save_name}")
#  python data/plot_graph_details.py   --excel checkpoints/grid_graphs/36spins/test_res_custom.xlsx   --data_root data --data_root data --idx 0
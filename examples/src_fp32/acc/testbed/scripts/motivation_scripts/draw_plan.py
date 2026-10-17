import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import from_pydot


# ----------- 解析 plan 函数 -----------
def parse_plan(plan_text: str):
    """
    解析 plan 文本生成 plan_ops 列表 (op_name, order, stage_id)
    """
    plan_text = plan_text.strip()

    # 按阶段切分
    stages = plan_text.split("|")

    plan_ops = []
    order = 1
    stage_id = 0

    for stage in stages:
        stage = stage.strip()
        # 去掉 (1)> 等标识
        if ">" in stage:
            stage = stage.split(">")[1].strip()

        # 按逗号切分
        ops = [op.strip() for op in stage.split(",") if op.strip()]

        # 如果第一个有数字前缀（比如 "45,G2sSparseIndexLoadOp"），只取最后部分
        if ops and ops[0][0].isdigit():
            print(ops[0])
            # ops[0] = ops[0].split(",")[-1].strip()
            plan_number = (int)(ops[0])
            ops = ops[1:]
            print(ops[0])

        # 生成 plan_ops
        for op in ops:
            plan_ops.append((op, order, stage_id))
            order += 1

        stage_id += 1

    return plan_number, plan_ops



def parse_plan_draw_fig(plan_text):
    plan_number, plan_ops = parse_plan(plan_text)

    # 依赖关系 (from -> to)
    dependencies = [
        ("G2sSparseIndexLoadOp", "G2sMatrixBLoadOp"),
        ("G2sMatrixBLoadOp", "S2rBValLoadOp"),
        # ("G2rSparseMcoOffLoadOp", "G2rSparseMcoMaskLoadOp"),
        ("G2rSparseMcoOffLoadOp", "G2sSparseMcoValLoadOp"),
        ("G2rSparseMcoMaskLoadOp", "S2sRestoreMatrixAOp"),
        ("G2sSparseMcoValLoadOp", "S2sRestoreMatrixAOp"),
        ("S2sRestoreMatrixAOp", "S2rAValLoadOp"),
        ("S2rAValLoadOp", "CalculateOp"),
        ("S2rBValLoadOp", "CalculateOp"),
    ]

    plan_latency={
        "G2sSparseIndexLoadOp": 12,
        "G2rSparseMcoOffLoadOp": 2,
        "G2rSparseMcoMaskLoadOp": 1,
        "G2sSparseMcoValLoadOp": 28,
        "G2sMatrixBLoadOp": 82,
        "S2sRestoreMatrixAOp": 42,
        "S2rAValLoadOp": 1,
        "S2rBValLoadOp": 2,
        "CalculateOp": 2
    }

    # ----------- 构建图 -----------
    G = nx.DiGraph()

    # 阶段颜色
    stage_colors = {0: "#ff9999", 1: "#99ccff", 2: "#99ff99"}

    # 为 op 生成带编号的名字映射
    op_name_map = {op: f"({order}) {op} ({plan_latency[op]})" for op, order, stage in plan_ops}

    # 添加节点
    for op, order, stage in plan_ops:
        G.add_node(op_name_map[op], color=stage_colors[stage])

    # 添加边 (需要映射到带编号的名字)
    for src, dst in dependencies:
        G.add_edge(op_name_map[src], op_name_map[dst])

    # ----------- 绘制 -----------
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G, prog="dot", args=" -Gnodesep=0.1 -Granksep=1")  # 使用 Graphviz 层次布局
    # pos = nx.nx_pydot.graphviz_layout(G, prog="dot")


    
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    return G, pos, node_colors, plan_number

def draw_pic(G, pos, node_colors, plan_number, i, plan_fig_dir, time_dir):
    plt.figure(figsize=(16, 8), dpi=100)
    nx.draw(G, pos, with_labels=True,
            node_color=node_colors, node_size=4000,
            font_size=8, font_color="black", arrowsize=15)

    plt.title("Op Execution Dependency Graph", fontsize=12)
    plt.tight_layout()
    # plt.savefig(f"/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_fig/plan_{plan_number}.jpg")
    plt.savefig(plan_fig_dir + time_dir + f"no_{i}_plan_{plan_number}.jpg")


# ----------- 输入 plan 文本 -----------
plan_text = "45,G2sSparseIndexLoadOp, G2rSparseMcoOffLoadOp |(1)> G2rSparseMcoMaskLoadOp, G2sSparseMcoValLoadOp, G2sMatrixBLoadOp |(1)> S2sRestoreMatrixAOp, S2rAValLoadOp, S2rBValLoadOp, CalculateOp"

all_plan_filepath="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/filtered_plans.txt"
all_plan = {}
with open(all_plan_filepath, "r") as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith("#"):  # 忽略空行或注释
            continue
        if "," in line:
            first_comma = line.find(",")
            plan_number = line[:first_comma].strip()
            all_plan[plan_number] = line
        else:
            plan_number = None

plan_fig_dir="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_fig/"

for i in range(11, 23):
    base = i // 10
    last = i % 10
    time_dir  = str(base) + "_" + str(last) + "ms/"
    plan_path = plan_fig_dir + time_dir + "plans.txt"


    # draw_fast_plan_fig_path="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/less_1.4ms_plan.txt"

    with open(plan_path, "r") as f:
        i = 0
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):  # 忽略空行或注释
                continue
            i = i + 1
            G, pos, node_colors, plan_number = parse_plan_draw_fig(all_plan[line])
            draw_pic(G, pos, node_colors, plan_number, i, plan_fig_dir, time_dir)
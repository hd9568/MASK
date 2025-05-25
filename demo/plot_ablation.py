import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 设置字体
# plt.rcParams["font.family"] = [
#     "Times New Roman",
#     "Songti SC",
# ]  # 英文用 Times New Roman，中文用宋体-简
# plt.rcParams["mathtext.fontset"] = "stix"  # 数学字体也使用 Times New Roman
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 为每个程序分配固定的颜色
def get_program_colors():
    colors = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]
    return colors


# 定义变体名称映射
variant_map = {
    "nopipe-nopre-noswz": "nopipe+nopre+noswz",
    "nopipe-nopre": "nopipe+nopre",
    "nopipe": "nopipe",
    "opt": "all",
}

# 定义算法名称映射
algorithm_map = {
    "dtc": "ME-TCF",
    "acc": "BitTCF",
    "bitbsr": "BitBSR",
    "srbcrs": "SR-BCRS",
    "cusp": "cuSPARSE"
}

def plot_grouped_bar_chart(plot_name, json_data, N, font_scaler=1.0):
    # 设置基础字体大小
    base_fontsize = 10
    scaled_fontsize = base_fontsize * font_scaler
    legend_fontsize = base_fontsize * font_scaler * 0.9

    # 准备数据
    algorithms = ["dtc", "acc", "bitbsr", "srbcrs", "cusp"]
    variants = ["nopipe-nopre-noswz", "nopipe-nopre", "nopipe", "opt"]
    
    # 初始化数据矩阵
    data = np.zeros((len(variants), len(algorithms)))
    
    # 从JSON数据中提取值
    for i, algo in enumerate(algorithms):
        if algo == "cusp":
            # cuSPARSE只有一个值
            data[3, i] = json_data["A800"]["fp16"]["1024_1024_0.9.mtx"][f"{N}"][algo][0]
        else:
            for j, variant in enumerate(variants):
                key = f"{algo}-{variant}" if variant != "opt" else algo
                data[j, i] = json_data["A800"]["fp16"]["1024_1024_0.9.mtx"][f"{N}"][key][0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 获取自定义颜色
    colors = get_program_colors()

    # 计算位置
    width = 0.8 / len(variants)
    x = np.zeros(len(algorithms))
    x[:-1] = np.arange(len(algorithms)-1)  # 前四个算法的位置
    x[-1] = len(algorithms)-1 - 0.3  # cuSPARSE的位置，向左偏移以减小间距

    # 绘制每组柱子
    for i, variant in enumerate(variants):
        if i == 3:  # 对于opt变体
            # 为所有算法绘制柱子
            if variant == "opt":  # 对于cuSPARSE的all变体
                ax.bar(x[-1] + 1.5 * width, data[i, -1], width, label=variant_map[variant], color=colors[4])
                ax.hlines(y=data[i, -1], xmin=-0.5, xmax=len(algorithms)-0.5, colors=colors[4], linestyles='dashed')
                ax.bar(x[:-1] + i * width, data[i, :-1], width, color=colors[i])
            else:
                ax.bar(x + i * width, data[i], width, label=variant_map[variant], color=colors[i])
        else:
            # 为除cuSPARSE外的算法绘制柱子
            ax.bar(x[:-1] + i * width, data[i, :-1], width, label=variant_map[variant], color=colors[i])

    # 设置x轴标签
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([algorithm_map[algo] for algo in algorithms], fontsize=scaled_fontsize)

    # 添加图例
    ax.legend(fontsize=legend_fontsize, ncol=4, bbox_to_anchor=(0.5, 1.15), loc='center')

    # 添加标题和标签
    ax.set_ylabel("Time (ms)", fontsize=scaled_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=scaled_fontsize)

    # 自动调整布局
    plt.tight_layout()

    plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", "-i", type=str, default="log.A800.json")
    parser.add_argument("--N", "-N", type=int, default=1024)
    return parser.parse_args()

def main():
    args = parse_args()
    # 读取JSON数据
    with open(args.log_path, "r") as f:
        json_data = json.load(f)

    plot_name = args.log_path.split(".")[0]

    # 绘制图表
    plot_grouped_bar_chart(plot_name, json_data, args.N, font_scaler=1.5)


if __name__ == "__main__":
    main()

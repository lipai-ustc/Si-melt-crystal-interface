import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 画单晶面熵分布图（严格参考原画图代码）
# --------------------------------------------------
def plot_single_face_entropy(centers, mean_entropy, std_entropy, face_label, out_path,
                             x_min_plot=None, x_max_plot=None,
                             y_min=None, y_max=None,
                             color='C0'):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23  # 字体大小
    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Entropy (eV/atom)", fontsize=fs)

    # 截取 x 范围
    mask = np.ones_like(centers, dtype=bool)
    if x_min_plot is not None:
        mask &= (centers >= x_min_plot)
    if x_max_plot is not None:
        mask &= (centers <= x_max_plot)

    centers_plot = centers[mask]
    mean_entropy_plot = mean_entropy[mask]
    std_entropy_plot = std_entropy[mask]

    # 阴影 ±σ
    ax.fill_between(centers_plot,
                    mean_entropy_plot - std_entropy_plot,
                    mean_entropy_plot + std_entropy_plot,
                    color=color, alpha=0.2, linewidth=0)

    # 均值曲线
    ax.plot(centers_plot, mean_entropy_plot, color=color, lw=2, alpha=1)

    # y 轴刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.15))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))

    # x, y 轴范围
    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(-1.68, -0.85)  # 原画图代码固定范围

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")


# --------------------------------------------------
# 主函数
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\0_all_newpot-2"
    faces = ["100", "110", "111"]  # 晶面列表
    output_dir = os.path.join(base_dir, "entropy_replot")
    os.makedirs(output_dir, exist_ok=True)

    rel_range = 20  # 横坐标范围
    color = 'C0'

    for face in faces:
        csv_path = os.path.join(base_dir, face, "entropy", "plot",
                                "avg_smoothed_entropy_with_std_relative_to_interface.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        # 读取 CSV，列顺序：rel_grid, Smoothed_Entropy, Std_Entropy
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        centers = data[:, 0]
        mean_entropy = data[:, 1]
        std_entropy = data[:, 2]

        plot_file = os.path.join(output_dir, f"{face}_entropy_plot.png")
        plot_single_face_entropy(
            centers, mean_entropy, std_entropy, face,
            plot_file,
            x_min_plot=-rel_range,
            x_max_plot=rel_range,
            color=color
        )

if __name__ == "__main__":
    main()

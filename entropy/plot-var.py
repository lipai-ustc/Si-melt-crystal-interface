import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 画单晶面熵分布图（含 ±σ 阴影）
# --------------------------------------------------
def plot_single_face_entropy(centers, mean_entropy, std_entropy, face_label, out_path,
                             xlabel="z (Å)", ylabel="Entropy (eV/atom)",
                             x_min_plot=None, x_max_plot=None,
                             y_min=None, y_max=None, color='C0'):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23  # 字体大小参考第二段代码
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)

    # 截取 x 范围
    mask = np.ones_like(centers, dtype=bool)
    if x_min_plot is not None:
        mask &= (centers >= x_min_plot)
    if x_max_plot is not None:
        mask &= (centers <= x_max_plot)
    centers = centers[mask]
    mean_entropy = mean_entropy[mask]
    std_entropy = std_entropy[mask]

    # 阴影 ±σ （保留原始代码）
    ax.fill_between(centers,
                    mean_entropy - std_entropy,
                    mean_entropy + std_entropy,
                    color=color, alpha=0.3, linewidth=0)
    # 均值曲线
    ax.plot(centers, mean_entropy, color=color, lw=2, label=face_label, alpha=0.9)

    # y 轴刻度参考第二段代码
    ax.yaxis.set_major_locator(MultipleLocator(0.15))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)
    # y 轴范围参考第二段代码
    ax.set_ylim(-1.55, -0.98)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")

# --------------------------------------------------
# 主函数：对每个晶面单独绘图，颜色统一
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\0_all_newpot-2"
    faces = ["100", "110", "111"]
    output_dir = os.path.join(base_dir, "entropy_separate_plots")
    os.makedirs(output_dir, exist_ok=True)

    rel_range = 20  # 使用与第二段代码一致的范围设置
    color = 'C0'

    for face in faces:
        csv_path = os.path.join(base_dir, face, "entropy", "plot",
                                "avg_std_smoothed_entropy_relative_to_interface.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

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

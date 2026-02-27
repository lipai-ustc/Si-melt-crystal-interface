import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 画单个 lq6 分布的函数（均值线 + 无边框阴影，无 legend）
# --------------------------------------------------
def plot_and_save_smoothed_lq6_with_var(
        centers,
        mean_lq6,
        variance,
        out_path,
        xlabel="z (Å)",
        ylabel=r"$\mathrm{\overline{q}}_6$",
        x_min_plot=None,
        x_max_plot=None
    ):
    fig, ax = plt.subplots(figsize=(8, 5))

    fs = 23
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)

    # 可选截取 x 范围
    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(centers, dtype=bool)
        if x_min_plot is not None:
            mask &= (centers >= x_min_plot)
        if x_max_plot is not None:
            mask &= (centers <= x_max_plot)
        centers   = centers[mask]
        mean_lq6  = mean_lq6[mask]
        variance  = variance[mask]

    std_lq6 = np.sqrt(variance)

    # 阴影区间 mean ± std（不画边界线）
    ax.fill_between(
        centers,
        mean_lq6 - std_lq6,
        mean_lq6 + std_lq6,
        color='C0',
        alpha=0.3,
        linewidth=0
    )

    # 保留均值曲线
    ax.plot(centers, mean_lq6, color='C0', lw=2, alpha=0.9)

    # y 轴刻度
    ax.set_ylim(0.2, 0.84)
    ax.yaxis.set_major_locator(MultipleLocator(0.16))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] LQ6 阴影图已保存: {out_path}")


# --------------------------------------------------
# 主函数：读取 CSV 文件直接画图
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\MLSCAN\python_script\analysis\lq6-interface"
    rel_range = 20

    csv_files = {
        "100": os.path.join(base_dir, "100", "avg_var_smoothed_lq6_relative_to_interface.csv"),
        "110": os.path.join(base_dir, "110", "avg_var_smoothed_lq6_relative_to_interface.csv"),
        "111": os.path.join(base_dir, "111", "avg_var_smoothed_lq6_relative_to_interface.csv"),
    }

    for face, csv_path in csv_files.items():
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        centers   = data[:, 0]
        mean_lq6  = data[:, 1]
        variance  = data[:, 2]

        output_plot = os.path.join(os.path.dirname(csv_path), f"{face}_lq6_plot_with_var.png")

        plot_and_save_smoothed_lq6_with_var(
            centers,
            mean_lq6,
            variance,
            out_path=output_plot,
            x_min_plot=-rel_range,
            x_max_plot=rel_range
        )

if __name__ == "__main__":
    main()

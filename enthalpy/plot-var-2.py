import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 画单个晶面平滑能量的函数（均值线 + 阴影 ±σ，无 legend）
# --------------------------------------------------
def plot_and_save_smoothed_energy_with_std(
        centers,
        mean_energy,
        std_energy,
        out_path,
        xlabel="z (Å)",
        ylabel="Enthalpy (eV/atom)",
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
        centers = centers[mask]
        mean_energy = mean_energy[mask]
        std_energy = std_energy[mask]

    # 阴影区间 mean ± std（不画边界线）
    ax.fill_between(
        centers,
        mean_energy - std_energy,
        mean_energy + std_energy,
        color='C0',
        alpha=0.2,
        linewidth=0
    )

    # 均值曲线
    ax.plot(centers, mean_energy, color='C0', lw=2, alpha=1)

    # y 轴范围和刻度
    ax.set_ylim(-10, -9.15)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] 平滑能量图已保存: {out_path}")


# --------------------------------------------------
# 主函数：读取计算代码生成的合并后 CSV 文件并画图
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\0_all_newpot-2"
    rel_range = 20

    csv_files = {
        "100": os.path.join(base_dir, "100", "enthalpy", "avg_std_smoothed_energy_relative_to_interface_merged.csv"),
        "110": os.path.join(base_dir, "110", "enthalpy", "avg_std_smoothed_energy_relative_to_interface_merged.csv"),
        "111": os.path.join(base_dir, "111", "enthalpy", "avg_std_smoothed_energy_relative_to_interface_merged.csv"),
    }

    for face, csv_path in csv_files.items():
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        # 读取 CSV：列名依次是 Relative_x, Mean_Smoothed_Energy, Std_Energy_per_slice
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        centers = data[:, 0]
        mean_energy = data[:, 1]
        std_energy = data[:, 2]

        output_plot = os.path.join(os.path.dirname(csv_path), f"{face}_smoothed_energy_merged_plot.png")

        plot_and_save_smoothed_energy_with_std(
            centers,
            mean_energy,
            std_energy,
            out_path=output_plot,
            x_min_plot=-rel_range,
            x_max_plot=rel_range
        )


if __name__ == "__main__":
    main()

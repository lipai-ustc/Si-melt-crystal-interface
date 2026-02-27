import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 画单个 lq6 分布的函数
# --------------------------------------------------
def plot_and_save_smoothed_lq6(
        centers,
        lq6,
        out_path,
        xlabel="z (Å)",
        ylabel=r"$\mathrm{\overline{q}}_6$",
        x_min_plot=None,
        x_max_plot=None
    ):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

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
        lq6 = lq6[mask]

    ax.plot(centers, lq6, color='C0', lw=2, alpha=0.9)

    # 虚线标记界面（如果不需要可注释）
    #ax.axvline(-3.3, color='r', ls='--', lw=1)
    #ax.axvline( 3.4, color='r', ls='--', lw=1)

    # y 轴刻度（根据 lq6 范围调整）
    ax.set_ylim(0.2, 0.84)
    ax.yaxis.set_major_locator(MultipleLocator(0.16))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)



    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] LQ6 Plot saved to: {out_path}")

# --------------------------------------------------
# 主函数：分别绘制 100、110、111 三张图
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\MLSCAN\python_script\analysis\lq6-interface"
    rel_range = 20

    # 三个晶面的 csv 文件
    csv_files = {
        "100": os.path.join(base_dir, "100", "avg_smoothed_lq6_relative_to_interface.csv"),
        "110": os.path.join(base_dir, "110", "avg_smoothed_lq6_relative_to_interface.csv"),
        "111": os.path.join(base_dir, "111", "avg_smoothed_lq6_relative_to_interface.csv"),
    }

    for face, csv_path in csv_files.items():
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        centers = data[:, 0]
        lq6     = data[:, 1]

        # 输出文件保存在对应的晶面文件夹
        output_plot = os.path.join(os.path.dirname(csv_path), f"{face}_lq6_plot.png")

        plot_and_save_smoothed_lq6(
            centers,
            lq6,
            out_path=output_plot,
            x_min_plot=-rel_range,
            x_max_plot=rel_range
        )

if __name__ == "__main__":
    main()

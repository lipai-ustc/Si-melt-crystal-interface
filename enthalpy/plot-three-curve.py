import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------
# 1. 画图函数：支持多条曲线
# --------------------------------------------------
def plot_and_save_smoothed_energy(
        curves,          # 传入一个 dict/list，每个元素形如 (centers, energy, label, color)
        out_path,
        sigma=None,      # 这里 sigma 不再用到，保留接口
        xlabel="z (Å)",
        ylabel="Enthalpy (eV/atom)",
        x_min_plot=None,
        x_max_plot=None
    ):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)

    for centers, energy, label, color in curves:
        # 可选截取 x 范围
        if x_min_plot is not None or x_max_plot is not None:
            mask = np.ones_like(centers, dtype=bool)
            if x_min_plot is not None:
                mask &= (centers >= x_min_plot)
            if x_max_plot is not None:
                mask &= (centers <= x_max_plot)
            centers = centers[mask]
            energy    = energy[mask]

        ax.plot(centers, energy, color=color, lw=2, label=label, alpha=0.9)

   # ax.axvline(-3.29999999999947, color='r', ls='--', lw=1)
  #  ax.axvline(3.40000000000061, color='r', ls='--', lw=1)
    # y 轴刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.15))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(-9.85, -9.25)

    #ax.legend(fontsize=fs-4, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")

# --------------------------------------------------
# 2. 主函数：一次性读 100 110 111
# --------------------------------------------------
def main():
    base_dir = r"E:\free_energy\0_all_newpot-2"
    output_dir = os.path.join(base_dir, "plot-enthalpy100")
    os.makedirs(output_dir, exist_ok=True)

    rel_range = 20

    # 三个晶面的 csv 文件
    csv_files = {
        "100": os.path.join(base_dir, "100", "enthalpy", "100.csv"),
        #"110": os.path.join(base_dir, "110", "enthalpy", "110.csv"),
        #"111": os.path.join(base_dir, "111", "enthalpy", "111.csv"),
    }

    # 颜色映射
   # colors = {"100": 'C0', "110": 'C1', "111": 'C2'}
    colors = {"100": 'C0'}
    curves = []
    for face, csv_path in csv_files.items():
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        centers = data[:, 0]
        energy  = data[:, 1]
        curves.append((centers, energy, f"{face}", colors[face]))

    # 绘图
    plot_file = os.path.join(output_dir, "multi_face_energy_plot.png")
    plot_and_save_smoothed_energy(
        curves,
        out_path=plot_file,
        x_min_plot=-rel_range,
        x_max_plot=rel_range
    )

if __name__ == "__main__":
    main()
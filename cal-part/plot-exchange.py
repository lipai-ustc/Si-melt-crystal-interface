import numpy as np
import matplotlib.pyplot as plt
import os


def plot_lq6_vs_x_colored_by_enthalpy(x_atoms, energies, lq6_values, x_bins, x_center, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))

    # 转换为相对坐标
    x_rel = x_atoms - x_center
    x_bins_rel = x_bins - x_center

    # 手动设置颜色范围
    vmin, vmax = -10.0, -9.0

    for i in range(len(x_bins_rel) - 1):
        mask = (x_rel >= x_bins_rel[i]) & (x_rel < x_bins_rel[i + 1])
        if np.any(mask):
            ax.scatter(x_rel[mask], lq6_values[mask],
                       c=energies[mask], cmap='viridis', s=8,
                       vmin=vmin, vmax=vmax, edgecolor='none')

    ax.set_xlim(x_bins_rel[0], x_bins_rel[-1])
    ax.set_ylim(-0.2, 1.0)  # 手动设置lq6范围
    ax.set_xlabel("x relative to center (Å)", fontsize=16)
    ax.set_ylabel("lq6", fontsize=16)
    ax.set_title("Per-atom lq6 vs relative x (colored by Enthalpy)", fontsize=16)

    # 设置刻度：以中心为0，±10、±20
    xticks = np.arange(-20, 21, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:+.0f}" for x in xticks])

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Enthalpy (eV/atom)", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 散点图已保存: {out_path}")


def main():
    txt_file = r"E:\free_energy\0all_new_pot\111\trypart_energy_lq6color\atom_data_frames_193-194.txt"
    output_dir = r"E:\free_energy\0all_new_pot\111\trypart_energy_lq6color"
    os.makedirs(output_dir, exist_ok=True)

    data = np.loadtxt(txt_file, comments="#")
    x_atoms, energies, lq6_values = data[:, 0], data[:, 1], data[:, 2]

    x_min, x_max = 63.53, 103.53
    dx = 0.1
    x_bins = np.arange(x_min, x_max + dx, dx)

    x_center = (x_min + x_max) / 2

    plot_file = os.path.join(output_dir, "exchange.png")
    plot_lq6_vs_x_colored_by_enthalpy(x_atoms, energies, lq6_values, x_bins, x_center, plot_file)


if __name__ == "__main__":
    main()

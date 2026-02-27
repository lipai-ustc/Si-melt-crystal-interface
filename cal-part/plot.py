import numpy as np
import matplotlib.pyplot as plt
import os


def plot_energy_vs_x_colored_by_lq6(x_atoms, energies, lq6_values, x_bins, x_center, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))

    # 转换为相对坐标
    x_rel = x_atoms - x_center
    x_bins_rel = x_bins - x_center

    vmin, vmax = np.nanmin(lq6_values), np.nanmax(lq6_values)

    for i in range(len(x_bins_rel) - 1):
        mask = (x_rel >= x_bins_rel[i]) & (x_rel < x_bins_rel[i + 1])
        if np.any(mask):
            ax.scatter(x_rel[mask], energies[mask],
                       c=lq6_values[mask], cmap='viridis', s=8,
                       vmin=vmin, vmax=vmax, edgecolor='none')

    ax.set_xlim(x_bins_rel[0], x_bins_rel[-1])
    ax.set_xlabel("x relative to center (Å)", fontsize=16)
    ax.set_ylabel("Enthalpy (eV/atom)", fontsize=16)
    ax.set_title("Per-atom Enthalpy vs relative x (colored by lq6)", fontsize=16)

    # 设置刻度：以中心为0，±10、±20
    xticks = np.arange(-20, 21, 10)  # 可根据需要调整范围
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:+.0f}" for x in xticks])  # 显示 +10/-10 这种格式

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("lq6", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 散点图已保存: {out_path}")


def main():
    txt_file = r"E:\free_energy\0all_new_pot\110\trypart_energy_lq6color\atom_data_frames_188-189.txt"
    output_dir = r"E:\free_energy\0all_new_pot\110\trypart_energy_lq6color"
    os.makedirs(output_dir, exist_ok=True)

    data = np.loadtxt(txt_file, comments="#")
    x_atoms, energies, lq6_values = data[:, 0], data[:, 1], data[:, 2]

    x_min, x_max = 67.434, 107.434
    dx = 0.1
    x_bins = np.arange(x_min, x_max + dx, dx)

    # 计算中点
    x_center = (x_min + x_max) / 2

    plot_file = os.path.join(output_dir, "atomwise_energy_lq6color_from_txt_relx.png")
    plot_energy_vs_x_colored_by_lq6(x_atoms, energies, lq6_values, x_bins, x_center, plot_file)


if __name__ == "__main__":
    main()

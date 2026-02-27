import numpy as np
import os
import matplotlib.pyplot as plt

# ====== Step 1: 读取 .xyz 文件（含 lq6 和能量）======
def load_xyz_with_lq6_energy(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            try:
                num_atoms = int(f.readline())
                comment = f.readline()
                atoms = []
                for _ in range(num_atoms):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    x, y, z, lq6, entropy, energy = map(float, parts[1:])
                    atoms.append((x, y, z, lq6, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames

# ====== Step 2: 画 x vs enthalpy 的散点图（颜色显示 lq6）======
def plot_energy_vs_x_colored_by_lq6(x_atoms, energies, lq6_values, x_bins, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))

    # 用 lq6 作为颜色
    vmin, vmax = np.nanmin(lq6_values), np.nanmax(lq6_values)

    for i in range(len(x_bins) - 1):
        mask = (x_atoms >= x_bins[i]) & (x_atoms < x_bins[i + 1])
        if np.any(mask):
            ax.scatter(x_atoms[mask], energies[mask],
                       c=lq6_values[mask], cmap='viridis', s=8,
                       vmin=vmin, vmax=vmax, edgecolor='none')

    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_xlabel("x (Å)", fontsize=16)
    ax.set_ylabel("Enthalpy (eV/atom)", fontsize=16)
    ax.set_title("Per-atom Enthalpy vs x (colored by lq6)", fontsize=16)

    # 颜色条显示 lq6 分布
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("lq6", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 每个原子的 enthalpy (颜色显示 lq6) 散点图已保存: {out_path}")


# ====== 主程序 ======
def main():
    xyz_file = r"E:\free_energy\0all_new_pot\111\cal_result\merge.xyz"
    output_dir = r"E:\free_energy\0all_new_pot\111\trypart_energy_lq6color"
    os.makedirs(output_dir, exist_ok=True)

    x_min, x_max = 63.53, 103.53
    dx = 0.1
    x_bins = np.arange(x_min, x_max + dx, dx)

    y_min, y_max = 27,40
    z_min, z_max = 15,37

    frame_start, frame_end = 193,194

    frames = load_xyz_with_lq6_energy(xyz_file)

    all_x, all_e, all_lq6 = [], [], []

    for i, frame in enumerate(frames):
        if i < frame_start or i > frame_end:
            continue

        x_atoms = frame[:, 0]
        y_atoms = frame[:, 1]
        z_atoms = frame[:, 2]
        lq6_values = frame[:, 3]
        energies = frame[:, 4]

        mask = (y_atoms > y_min) & (y_atoms < y_max) & (z_atoms > z_min) & (z_atoms < z_max)
        if np.sum(mask) == 0:
            continue

        all_x.append(x_atoms[mask])
        all_e.append(energies[mask])
        all_lq6.append(lq6_values[mask])

    if len(all_x) == 0:
        print("❌ 没有符合条件的原子，无法绘制散点图")
        return

    all_x = np.concatenate(all_x)
    all_e = np.concatenate(all_e)
    all_lq6 = np.concatenate(all_lq6)

    plot_file = os.path.join(output_dir, f"atomwise_energy_lq6color_frames_{frame_start}-{frame_end}.png")
    plot_energy_vs_x_colored_by_lq6(all_x, all_e, all_lq6, x_bins, plot_file)

if __name__ == "__main__":
    main()

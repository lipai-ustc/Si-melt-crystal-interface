import numpy as np
import os
import matplotlib.pyplot as plt

# ====== Step 1: 读取 .xyz 文件（含能量）======
def load_xyz_with_energy(filename):
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
                    # 读取 x, y, z, energy
                    x, y, z, lq6, entropy, energy = map(float, parts[1:])
                    atoms.append((x, y, z, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames

# ====== Step 2: 高斯平滑函数 (能量)======
def gaussian_smooth_energy(x_atoms, energies, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * energies) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

# ====== Step 3: 可视化平均能量分布 ======
def plot_avg_energy(x_grid, avg_energy, out_path, x_min, x_max):
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, avg_energy, color='C0')
    plt.xlabel("x (Å)", fontsize=16)
    plt.ylabel("Enthalpy (eV/atom)", fontsize=16)
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 平均能量分布图已保存: {out_path}")

# ====== 主程序 ======
def main():
    # === 参数设置 ===
    xyz_file = r"E:\free_energy\0all_new_pot\111\cal_result\merge.xyz"
    output_dir = r"E:\free_energy\0all_new_pot\111\trypart_energy"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    x_min, x_max = 60, 100  # 你可以改成实际系统的 x 范围
    dx = 0.1
    x_grid = np.arange(x_min, x_max + dx, dx)

    y_min, y_max = 22,34      #111 43, 60.8935  22,34
    z_min, z_max =  17, 29     #111 0, 18     17, 29

    frame_start, frame_end = 260, 261  # <=== 修改帧范围

    # === 读取数据 ===
    frames = load_xyz_with_energy(xyz_file)

    smoothed_selected_frames = []

    for i, frame in enumerate(frames):
        if i < frame_start or i > frame_end:
            continue  # 跳过不在范围内的帧

        x_atoms = frame[:, 0]
        y_atoms = frame[:, 1]
        z_atoms = frame[:, 2]
        energies = frame[:, 3]

        # 过滤 y,z 范围
        mask = (y_atoms > y_min) & (y_atoms < y_max) & (z_atoms > z_min) & (z_atoms < z_max)
        if np.sum(mask) == 0:
            continue  # 该帧没有符合条件的原子，跳过

        x_atoms_filtered = x_atoms[mask]
        energies_filtered = energies[mask]

        smoothed = gaussian_smooth_energy(x_atoms_filtered, energies_filtered, x_grid, sigma)
        smoothed_selected_frames.append(smoothed)

    smoothed_selected_frames = np.array(smoothed_selected_frames)
    avg_smoothed = np.nanmean(smoothed_selected_frames, axis=0)

    # === 保存结果 ===
    output_csv = os.path.join(output_dir, f"avg_smoothed_energy_frames_{frame_start}-{frame_end}.csv")
    np.savetxt(output_csv, np.column_stack([x_grid, avg_smoothed]),
               delimiter=',', header='x,Smoothed_Energy', comments='')
    print(f"✅ 平均能量分布已保存: {output_csv}")

    plot_file = os.path.join(output_dir, f"avg_smoothed_energy_plot_frames_{frame_start}-{frame_end}.png")
    plot_avg_energy(x_grid, avg_smoothed, plot_file, x_min, x_max)

if __name__ == "__main__":
    main()

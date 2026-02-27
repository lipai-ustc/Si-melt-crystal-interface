import numpy as np
import os
import matplotlib.pyplot as plt

# ====== Step 1: 读取 .xyz 文件 ======
def load_xyz_with_lq6(filename):
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
                    atoms.append((x, y, z, lq6, entropy, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames

# ====== Step 2: 读取 interface 文件 ======
def load_interfaces(interface_file):
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

# ====== Step 3: 高斯平滑函数 ======
def gaussian_smooth_lq6(x_atoms, lq6_values, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * lq6_values) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

# ====== Step 4: 计算每个slice的标准差 ======
def compute_std_per_slice(x_atoms, lq6_values, x_grid, dx):
    stds = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        mask = (x_atoms >= x0 - dx/2) & (x_atoms < x0 + dx/2)
        if np.any(mask):
            stds[i] = np.std(lq6_values[mask])
        else:
            stds[i] = np.nan
    return stds

# ====== 可视化 ======
def plot_lq6_with_std(rel_grid, avg_lq6, std_lq6, out_path, x_min, x_max):
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_lq6, color='C1', label='Mean lq6')
    plt.fill_between(rel_grid, avg_lq6-std_lq6, avg_lq6+std_lq6, color='C1', alpha=0.3, label='±1σ')
    plt.axvline(0, color='r', linestyle='--', label="Interface")
    plt.xlabel("Relative x (Å)", fontsize=16)
    plt.ylabel("lq6", fontsize=16)
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 平均 lq6 + 标准差分布图已保存: {out_path}")

# ====== 主程序 ======
def main():
    # === 参数设置 ===
    xyz_file = r"E:\free_energy\0all_new_pot\110\cal_result\merge.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\110\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\MLSCAN\python_script\analysis\lq6-interface\110"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    rel_range = 20
    dx = 0.1
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    # === 读取数据 ===
    frames = load_xyz_with_lq6(xyz_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与 interface.txt 不一致！"

    # === 把所有帧对齐后拼接 ===
    all_x = []
    all_lq6 = []
    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x  # 对齐到界面
        all_x.append(x_atoms)
        all_lq6.append(frame[:, 3])

    all_x = np.concatenate(all_x)
    all_lq6 = np.concatenate(all_lq6)

    # === 在合并后的数据上平滑 ===
    avg_smoothed = gaussian_smooth_lq6(all_x, all_lq6, rel_grid, sigma)
    std_per_slice = compute_std_per_slice(all_x, all_lq6, rel_grid, dx)

    # === 保存结果 ===
    output_csv = os.path.join(output_dir, "avg_smoothed_lq6_and_std_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed, std_per_slice]),
               delimiter=',', header='Relative_x,Smoothed_lq6,Std_lq6', comments='')
    print(f"✅ 平均对齐 lq6 + 标准差已保存: {output_csv}")

    plot_file = os.path.join(output_dir, "avg_smoothed_lq6_with_std_plot.png")
    plot_lq6_with_std(rel_grid, avg_smoothed, std_per_slice, plot_file, -rel_range, rel_range)

if __name__ == "__main__":
    main()

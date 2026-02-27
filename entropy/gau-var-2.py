import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 高斯平滑函数 ======
def gaussian_smooth_entropy(x_atoms, entropies, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * entropies) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

# ====== 按 slice 计算标准差 ======
def compute_std_per_slice(x_atoms, entropies, x_grid, dx):
    stds = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        left = x0 - dx/2
        right = x0 + dx/2
        mask = (x_atoms >= left) & (x_atoms < right)
        if np.any(mask):
            stds[i] = np.std(entropies[mask])
        else:
            stds[i] = np.nan
    return stds

# ====== 读取 entropy xyz 文件 ======
def read_all_frames_realentropy_xyz(filepath):
    """读取所有帧，返回列表，每个元素是 N x 5 的 numpy 数组 (x, y, z, id, entropy)"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header = lines[i + 1]
        atoms = lines[i + 2: i + 2 + num_atoms]
        data = []
        for line in atoms:
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            entropy = float(parts[4])
            atom_id = int(parts[0])
            data.append([x, y, z, atom_id, entropy])
        frames.append(np.array(data))
        i += 2 + num_atoms
    return frames

# ====== 读取 interface 文件 ======
def load_interfaces(interface_file):
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

# ====== 主程序 ======
def main():
    # === 参数设置 ===
    entropy_xyz_file = r"E:\free_energy\0_all_newpot-2\110\entropy\realentropy-TS.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\110\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\110\entropy\plot"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    dx = 0.1
    rel_range = 40
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    # === 读取并对齐所有帧 ===
    frames = read_all_frames_realentropy_xyz(entropy_xyz_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "帧数和 interface.txt 不一致"

    all_x = []
    all_entropy = []
    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x
        all_x.append(x_atoms)
        all_entropy.append(frame[:, 4])

    all_x = np.concatenate(all_x)
    all_entropy = np.concatenate(all_entropy)

    # === 在合并后的数据上平滑和计算标准差 ===
    avg_smoothed = gaussian_smooth_entropy(all_x, all_entropy, rel_grid, sigma)
    std_per_slice = compute_std_per_slice(all_x, all_entropy, rel_grid, dx)

    # === 保存 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_entropy_with_std_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed, std_per_slice]),
               delimiter=',', header='Relative_x,Smoothed_Entropy,Std_Entropy', comments='')
    print(f"✅ 平均熵 + 标准差 保存：{output_csv}")

    # === 可视化 ===
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_smoothed, color='C1', label=f"σ={sigma}")
    plt.fill_between(rel_grid, avg_smoothed-std_per_slice, avg_smoothed+std_per_slice,
                     color='C1', alpha=0.3, label='±1σ')
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Relative x (Å)")
    plt.ylabel("Smoothed Entropy (-TS, eV/atom)")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "avg_smoothed_entropy_with_std_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"✅ 图像保存：{plot_file}")

if __name__ == "__main__":
    main()

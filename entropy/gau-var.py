import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 高斯平滑熵
# --------------------------------------------------
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

# --------------------------------------------------
# 读取所有帧，返回列表，每帧 N x 5 (x,y,z,id,entropy)
# --------------------------------------------------
def read_all_frames_realentropy_xyz(filepath):
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
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

# --------------------------------------------------
# 读取界面位置
# --------------------------------------------------
def load_interfaces(interface_file):
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

# --------------------------------------------------
# 绘图函数（均值 + 阴影 ±σ）
# --------------------------------------------------
def plot_smoothed_entropy_with_std(centers, mean_entropy, std_entropy, out_path,
                                   xlabel="z (Å)", ylabel="Smoothed Entropy (-TS, eV/atom)",
                                   x_min_plot=None, x_max_plot=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    # 截取 x 范围
    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(centers, dtype=bool)
        if x_min_plot is not None:
            mask &= (centers >= x_min_plot)
        if x_max_plot is not None:
            mask &= (centers <= x_max_plot)
        centers = centers[mask]
        mean_entropy = mean_entropy[mask]
        std_entropy = std_entropy[mask]

    # 阴影 ±σ
    ax.fill_between(centers,
                    mean_entropy - std_entropy,
                    mean_entropy + std_entropy,
                    color='C1',
                    alpha=0.3,
                    linewidth=0)
    # 均值曲线
    ax.plot(centers, mean_entropy, color='C1', lw=2, alpha=0.9)
    ax.axvline(0, color='r', linestyle='--', label='Interface')

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] 熵分布图已保存: {out_path}")

# --------------------------------------------------
# 主函数
# --------------------------------------------------
def main():
    entropy_xyz_file = r"E:\free_energy\0_all_newpot-2\110\entropy\realentropy-TS.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\110\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\110\entropy\plot"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    dx = 0.1
    rel_range = 40

    # 读取数据
    frames = read_all_frames_realentropy_xyz(entropy_xyz_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "帧数和 interface.txt 不一致"

    rel_grid = np.arange(-rel_range, rel_range + dx, dx)
    smoothed_all_frames = []

    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x
        entropies = frame[:, 4]
        smoothed = gaussian_smooth_entropy(x_atoms, entropies, rel_grid, sigma)
        smoothed_all_frames.append(smoothed)

    smoothed_all_frames = np.array(smoothed_all_frames)

    # 计算均值和标准差
    mean_smoothed = np.nanmean(smoothed_all_frames, axis=0)
    std_smoothed = np.nanstd(smoothed_all_frames, axis=0)

    # 保存 CSV
    output_csv = os.path.join(output_dir, "avg_std_smoothed_entropy_relative_to_interface.csv")
    np.savetxt(output_csv,
               np.column_stack([rel_grid, mean_smoothed, std_smoothed]),
               delimiter=',',
               header='Relative_x,Mean_Smoothed_Entropy,Std_Smoothed_Entropy',
               comments='')
    print(f"✅ 平均值和标准差已保存: {output_csv}")

    # 绘图
    plot_file = os.path.join(output_dir, "avg_smoothed_entropy_plot.png")
    plot_smoothed_entropy_with_std(rel_grid, mean_smoothed, std_smoothed, plot_file,
                                   x_min_plot=-rel_range, x_max_plot=rel_range)

if __name__ == "__main__":
    main()

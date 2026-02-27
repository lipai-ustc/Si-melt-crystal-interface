import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 高斯平滑（与原来相同） ======
def gaussian_smooth_energy(x_atoms, energies, x_grid, sigma):
    """在给定的 x_grid 点上对 (x_atoms, energies) 做高斯加权平均，返回每个 grid 点的平滑值。"""
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * energies) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed


# ====== 读取 merge.xyz（与原来相同） ======
def read_all_frames_merge_xyz(filepath):
    """读取所有帧，返回列表，每个元素是 N x 4 的 numpy 数组 (x,y,z,energy)"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        # header = lines[i + 1]
        atoms = lines[i + 2: i + 2 + num_atoms]
        data = []
        for line in atoms:
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            energy = float(parts[6])
            data.append([x, y, z, energy])
        frames.append(np.array(data))
        i += 2 + num_atoms
    return frames


# ====== 读取 interface ======
def load_interfaces(interface_file):
    """读取 interface.txt，返回界面位置数组"""
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data


# ====== 计算每个 slice 的标准差（针对合并后的原子集合） ======
def compute_std_per_slice(x_atoms, energies, x_grid, dx):
    """对每个以 x_grid 为中心、宽度 dx 的 slice，计算该 slice 内原子 energies 的标准差。"""
    stds = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        left = x0 - dx / 2.0
        right = x0 + dx / 2.0
        mask = (x_atoms >= left) & (x_atoms < right)
        if np.any(mask):
            stds[i] = np.std(energies[mask])
        else:
            stds[i] = np.nan
    return stds


# ====== 绘图（与原来函数一致） ======
def plot_and_save_smoothed_energy(
        centers, mean_energy, std_energy, out_path,
        xlabel="z (Å)", ylabel="Enthalpy (eV/atom)",
        x_min_plot=None, x_max_plot=None
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
        centers     = centers[mask]
        mean_energy = mean_energy[mask]
        std_energy  = std_energy[mask]

    # 阴影区域 (mean ± std)
    ax.fill_between(
        centers,
        mean_energy - std_energy,
        mean_energy + std_energy,
        color='C0',
        alpha=0.3,
        linewidth=0
    )

    # 均值曲线
    ax.plot(centers, mean_energy, color='C0', lw=2)

    ax.axvline(0, color='r', linestyle='--')

    # 设置 y 轴范围和刻度（可按需调整）
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.12))
    ax.set_ylim(-9.88, -9.28)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] 平滑 Enthalpy 图已保存: {out_path}")


# ====== 主流程：先对齐并合并所有原子，再平滑并计算每个 slice 的标准差 ======
def main():
    # === 参数设置（请根据需要修改路径） ===
    merge_xyz_file = r"E:\free_energy\0all_new_pot\100\cal_result\merge.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\100\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\100\enthalpy"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    dx = 0.1
    rel_range = 20

    # === 读取所有帧和对应界面位置 ===
    frames = read_all_frames_merge_xyz(merge_xyz_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与 interface.txt 不一致"

    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    # === 把每帧对齐后合并所有原子（x 相对于界面） ===
    all_x = []
    all_energies = []
    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x
        energies = frame[:, 3]
        all_x.append(x_atoms)
        all_energies.append(energies)

    all_x = np.concatenate(all_x)
    all_energies = np.concatenate(all_energies)

    # === 在合并后的原子集合上进行一次高斯平滑（得到平均曲线） ===
    mean_smoothed = gaussian_smooth_energy(all_x, all_energies, rel_grid, sigma)

    # === 对每个 slice（宽度 dx）计算原始能量的标准差 ===
    std_per_slice = compute_std_per_slice(all_x, all_energies, rel_grid, dx)

    # === 保存 CSV，包含均值和平移后的标准差 ===
    output_csv = os.path.join(output_dir, "avg_std_smoothed_energy_relative_to_interface_merged.csv")
    np.savetxt(output_csv,
               np.column_stack([rel_grid, mean_smoothed, std_per_slice]),
               delimiter=',',
               header='Relative_x,Mean_Smoothed_Energy,Std_Energy_per_slice',
               comments='')
    print(f"✅ 合并后均值和 slice 标准差已保存: {output_csv}")

    # === 绘图（平均曲线 + ±1σ） ===
    plot_file = os.path.join(output_dir, "avg_smoothed_energy_merged_plot.png")
    plot_and_save_smoothed_energy(
        rel_grid, mean_smoothed, std_per_slice,
        out_path=plot_file,
        x_min_plot=-rel_range,
        x_max_plot=rel_range
    )

if __name__ == "__main__":
    main()

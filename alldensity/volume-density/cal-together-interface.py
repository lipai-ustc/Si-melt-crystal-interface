import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 读取 atomvolume.xyz 文件 =====
def load_atomvolume_xyz(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            _ = f.readline()  # 跳过注释行
            x_list = []
            vol_list = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                if len(parts) < 5:
                    continue
                x = float(parts[1])
                volume = float(parts[4])
                x_list.append(x)
                vol_list.append(volume)
            frames.append((np.array(x_list), np.array(vol_list)))
    return frames

# ===== 读取界面文件 =====
def load_interfaces(filename):
    data = np.loadtxt(filename)
    return data[:, 0]  # 假设第一列是界面位置

# ===== 高斯加权平滑 =====
def gaussian_smooth(x_atoms, values, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        smoothed[i] = np.sum(weights * values) / weight_sum if weight_sum > 0 else np.nan
    return smoothed

# ===== 分 bin 密度计算 =====
def binned_density(x_atoms, x_grid, Ly, Lz):
    bin_width = x_grid[1] - x_grid[0]
    bins = np.append(x_grid - bin_width/2, x_grid[-1]+bin_width/2)
    counts, _ = np.histogram(x_atoms, bins=bins)
    density = counts / (bin_width * Ly * Lz)
    return density

# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0all_new_pot\100\cal_result\atomvolume.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\100\cal_result\interface_55-bins.txt"
    output_path = r"E:\free_energy\0all_new_pot\100\density\correct"
    os.makedirs(output_path, exist_ok=True)

    # 盒子尺寸（Y、Z方向）
    Ly, Lz = 53.6126, 53.6126

    frames = load_atomvolume_xyz(atomvolume_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    # ---- 用户选择帧范围 ----
    start_frame = 10  # 起始帧（含）
    end_frame = 100  # 结束帧（不含）
    frames = frames[start_frame:end_frame]
    interfaces = interfaces[start_frame:end_frame]

    # 设置相对界面的位置网格
    rel_range = 20  # Å
    n_grid = 400
    rel_grid = np.linspace(-rel_range, rel_range, n_grid)

    sigmas = [0.8, 0.08]

    all_gaussian_dict = {sigma: [] for sigma in sigmas}
    all_bin_density = []

    # ===== 循环每帧 =====
    for (x_atoms, volumes), interface_x in zip(frames, interfaces):
        rel_x_atoms = x_atoms - interface_x  # 界面对齐
        densities = np.where(volumes > 0, 1.0 / volumes, 0.0)

        # 高斯加权密度
        for sigma in sigmas:
            smoothed = gaussian_smooth(rel_x_atoms, densities, rel_grid, sigma)
            all_gaussian_dict[sigma].append(smoothed)

        # 分 bin 密度
        bin_density = binned_density(rel_x_atoms, rel_grid, Ly, Lz)
        all_bin_density.append(bin_density)

    # ===== 平均 =====
    avg_gaussian = {sigma: np.mean(all_gaussian_dict[sigma], axis=0) for sigma in sigmas}
    avg_bin_density = np.mean(all_bin_density, axis=0)



    # ===== 保存数据 =====
    for sigma in sigmas:
        np.savetxt(
            os.path.join(output_path, f"gaussian_density_sigma{sigma}.txt"),
            np.column_stack((rel_grid, avg_gaussian[sigma])),
            header="Relative_x(A)    Gaussian density (1/Volume)",
            fmt="%.6f"
        )

    np.savetxt(
        os.path.join(output_path, "binned_density.txt"),
        np.column_stack((rel_grid, avg_bin_density)),
        header="Relative_x(A)    Binned density (atoms / bin volume)",
        fmt="%.6f"
    )

    # ===== 绘图 =====
    plt.figure(figsize=(10,6))
    colors = ['C0', 'C1']
    for i, sigma in enumerate(sigmas):
        plt.plot(rel_grid, avg_gaussian[sigma], label=f"Gaussian σ={sigma} Å", color=colors[i], linewidth=2)
    plt.plot(rel_grid, avg_bin_density, color='C2', linestyle='--', label="Binned density", linewidth=2)
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("x relative to interface (Å)", fontsize=12)
    plt.ylabel("Density (Å⁻³)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-rel_range, rel_range)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "density_comparison.png"), dpi=300)
    plt.close()

    print(f"处理完成，结果保存在 {output_path}")

if __name__ == "__main__":
    main()

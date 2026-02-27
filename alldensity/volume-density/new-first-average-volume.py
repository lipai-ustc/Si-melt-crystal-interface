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

# ===== 高斯平滑密度计算（不变） =====
def gaussian_smooth(x_atoms, values, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        smoothed[i] = np.sum(weights * values) / weight_sum if weight_sum > 0 else np.nan
    return smoothed

# ===== 分 bin 密度计算（按原子数 / bin体积） =====
def binned_density(x_atoms, x_grid, Ly, Lz):
    """
    x_atoms: 原子 x 坐标
    x_grid: bin 中心
    Ly, Lz: 盒子尺寸
    """
    bin_width = x_grid[1] - x_grid[0]
    bins = np.append(x_grid - bin_width / 2, x_grid[-1] + bin_width / 2)
    counts, _ = np.histogram(x_atoms, bins=bins)
    density = counts / (bin_width * Ly * Lz)  # 转为体密度
    return density

# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0dispot\100\result\atomvolume.xyz"
    output_path = r"E:\free_energy\0dispot\100\density\new-first-average"
    os.makedirs(output_path, exist_ok=True)

    # 盒子尺寸（y和z方向）
    Ly = 53.6126 # Å
    Lz = 53.6126 # Å

    # ---- 读取数据 ----
    frames = load_atomvolume_xyz(atomvolume_file)

    # ---- 用户选择帧范围 ----
    start_frame = 0
    end_frame = 20
    selected_frames = frames[start_frame:end_frame]

    # ---- 自动获取盒子 x 范围 ----
    all_x = np.concatenate([x_atoms for x_atoms, _ in selected_frames])
    x_min, x_max = all_x.min(), all_x.max()
    n_grid = 1000
    x_grid = np.linspace(x_min, x_max, n_grid)

    sigmas = [0.8, 0.08]

    all_gaussian_dict = {sigma: [] for sigma in sigmas}
    all_bin_density = []

    for x_atoms, volumes in selected_frames:
        # ---- 高斯平滑方法（不变） ----
        densities = np.where(volumes > 0, 1.0 / volumes, 0.0)
        for sigma in sigmas:
            smoothed = gaussian_smooth(x_atoms, densities, x_grid, sigma)
            all_gaussian_dict[sigma].append(smoothed)

        # ---- 分 bin 方法（原子数/bin体积） ----
        bin_density = binned_density(x_atoms, x_grid, Ly, Lz)
        all_bin_density.append(bin_density)

    # 平均
    avg_gaussian = {sigma: np.mean(all_gaussian_dict[sigma], axis=0) for sigma in sigmas}
    avg_bin_density = np.mean(all_bin_density, axis=0)

    # ===== 保存数据 =====
    for sigma in sigmas:
        np.savetxt(
            os.path.join(output_path, f"gaussian_sigma{sigma}.txt"),
            np.column_stack((x_grid, avg_gaussian[sigma])),
            header="x(Å)    Gaussian smoothed density (1/Voronoi)",
            fmt="%.6f",
            encoding='utf-8'
        )
    np.savetxt(
        os.path.join(output_path, "binned_density.txt"),
        np.column_stack((x_grid, avg_bin_density)),
        header="x(Å)    Binned density (atoms / bin volume)",
        fmt="%.6f",
        encoding='utf-8'
    )

    # ===== 绘图 =====
    plt.figure(figsize=(10,6))
    for sigma in sigmas:
        plt.plot(x_grid, avg_gaussian[sigma], label=f"Gaussian σ={sigma} Å", linewidth=2)
    plt.plot(x_grid, avg_bin_density, color='C1', linestyle='--', label="Binned density")
    plt.xlabel("x (Å)", fontsize=12)
    plt.ylabel("Density (Å⁻³)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "density_comparison.png"), dpi=300)
    plt.close()

    print(f"处理完成，结果保存在 {output_path}")
    print(f"x 范围: {x_min:.2f} Å 到 {x_max:.2f} Å")

if __name__ == "__main__":
    main()

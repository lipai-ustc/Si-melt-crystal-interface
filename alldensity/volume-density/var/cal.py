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

# ===== 读取 interface 文件 =====
def load_interfaces(filename):
    data = np.loadtxt(filename)
    return data[:, 0]

# ===== 高斯平滑密度计算 =====
def gaussian_smooth(x_atoms, values, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * values) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

# ===== 绘图函数（均值 + 阴影 ±σ） =====
def plot_density_with_std(rel_grid, mean_density, std_density, out_path,
                          xlabel="z (Å)", ylabel="Number density (Å⁻³)",
                          x_min_plot=None, x_max_plot=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    # 截取范围
    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(rel_grid, dtype=bool)
        if x_min_plot is not None:
            mask &= (rel_grid >= x_min_plot)
        if x_max_plot is not None:
            mask &= (rel_grid <= x_max_plot)
        rel_grid = rel_grid[mask]
        mean_density = mean_density[mask]
        std_density = std_density[mask]

    # 阴影 ±σ
    ax.fill_between(rel_grid,
                    mean_density - std_density,
                    mean_density + std_density,
                    color='C0',
                    alpha=0.3,
                    linewidth=0)
    # 均值曲线
    ax.plot(rel_grid, mean_density, color='C0', lw=2, alpha=0.9)
    ax.axvline(0, color='r', linestyle='--')
    ax.axvline(-10, color='r', linestyle='--')

    fs = 23
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_ylim(0.049, 0.058)

    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.002))

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] 平均密度 ±σ 图已保存: {out_path}")

# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0all_new_pot\111\cal_result\atomvolume.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\111\cal_result\interface_55.txt"
    output_path = r"E:\free_energy\0all_new_pot\111\density\aim-vor-density"
    os.makedirs(output_path, exist_ok=True)

    frames = load_atomvolume_xyz(atomvolume_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    # 设置相对界面位置网格
    rel_range = 20
    rel_grid = np.linspace(-rel_range, rel_range, 400)
    sigmas = [0.8, 0.08]

    # 存储每个 sigma 的所有帧平滑结果
    all_density_dict = {sigma: [] for sigma in sigmas}

    for (x_atoms, volumes), interface_x in zip(frames, interfaces):
        rel_x_atoms = x_atoms - interface_x
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(volumes != 0, 1.0 / volumes, 0.0)
        for sigma in sigmas:
            smoothed_density = gaussian_smooth(rel_x_atoms, densities, rel_grid, sigma)
            all_density_dict[sigma].append(smoothed_density)

    # 转换为 numpy 数组
    for sigma in sigmas:
        all_density_dict[sigma] = np.array(all_density_dict[sigma])

    # 计算平均值
    avg_densities = {sigma: np.nanmean(all_density_dict[sigma], axis=0) for sigma in sigmas}

    # 保存两个 sigma 的平均值 CSV
    for sigma in sigmas:
        np.savetxt(
            os.path.join(output_path, f"density_avg_sigma{sigma}.csv"),
            np.column_stack((rel_grid, avg_densities[sigma])),
            delimiter=',',
            header="Relative_x,Mean_Density",
            comments=''
        )
        print(f"✅ sigma={sigma} 平均值已保存")

    # sigma=0.8 时计算标准差并保存 CSV
    sigma_std = 0.8
    std_density = np.nanstd(all_density_dict[sigma_std], axis=0)
    output_csv_std = os.path.join(output_path, f"density_avg_std_sigma{sigma_std}.csv")
    np.savetxt(
        output_csv_std,
        np.column_stack((rel_grid, avg_densities[sigma_std], std_density)),
        delimiter=',',
        header="Relative_x,Mean_Density,Std_Density",
        comments=''
    )
    print(f"✅ sigma={sigma_std} 均值和标准差已保存: {output_csv_std}")

    # 绘图
    plot_file = os.path.join(output_path, f"density_sigma{sigma_std}_plot.png")
    plot_density_with_std(rel_grid, avg_densities[sigma_std], std_density, plot_file,
                          x_min_plot=-rel_range, x_max_plot=rel_range)

if __name__ == "__main__":
    main()

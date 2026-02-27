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

# ===== 高斯平滑密度计算（加权平均版） =====
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

# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0all_new_pot\111\cal_result\atomvolume.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\111\cal_result\interface_55.txt"
    output_path = r"E:\free_energy\0all_new_pot\111\density\aim-vor-density"
    os.makedirs(output_path, exist_ok=True)

    frames = load_atomvolume_xyz(atomvolume_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    # 设置相对界面的位置网格
    rel_range = 20  # Å
    rel_grid = np.linspace(-rel_range, rel_range, 400)

    sigmas = [0.8, 0.08]

    all_density_dict = {sigma: [] for sigma in sigmas}

    for (x_atoms, volumes), interface_x in zip(frames, interfaces):
        rel_x_atoms = x_atoms - interface_x

        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(volumes != 0, 1.0 / volumes, 0.0)

        for sigma in sigmas:
            smoothed_density = gaussian_smooth(rel_x_atoms, densities, rel_grid, sigma)
            all_density_dict[sigma].append(smoothed_density)

    avg_densities = {
        sigma: np.mean(all_density_list, axis=0)
        for sigma, all_density_list in all_density_dict.items()
    }

    # 保存两组数据
    for sigma in sigmas:
        np.savetxt(
            os.path.join(output_path, f"density_relative_to_interface_sigma{sigma}.txt"),
            np.column_stack((rel_grid, avg_densities[sigma])),
            header="Relative_x(A)    Smoothed AtomicDensity (1/Volume)", fmt="%.5f"
        )

    # ======== 绘图 ========
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23

    plot_x_min = -20
    plot_x_max = 20

    mask = (rel_grid >= plot_x_min) & (rel_grid <= plot_x_max)
    rel_grid_plot = rel_grid[mask]

    # 统一颜色，不同透明度
    line_color = 'C0'
    alphas = {0.8: 0.9, 0.08: 0.2}

    for sigma in sigmas:
        avg_density_plot = avg_densities[sigma][mask]
        ax.plot(
            rel_grid_plot,
            avg_density_plot,
            color=line_color,
            alpha=alphas[sigma],
            label=f"σ={sigma} Å"
        )

    # 红色虚线，没有 label
    ax.axvline(0, color='r', linestyle='--')
    ax.axvline(-10, color='r', linestyle='--')


    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Number density (Å⁻³)", fontsize=fs)

    ax.set_ylim(0.049, 0.058)

    from matplotlib.pyplot import MultipleLocator
    y_major_locator = MultipleLocator(0.002)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_xlim(plot_x_min, plot_x_max)

    ax.grid(False)
    ax.legend(fontsize=fs - 4)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "all_frames_sigma_0.5_0.2_density.png"),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.01
    )
    plt.close()
    print("rel_grid_plot min:", rel_grid_plot.min())
    print("rel_grid_plot max:", rel_grid_plot.max())
    print(f"处理完成，结果保存在 {output_path}")

if __name__ == "__main__":
    main()

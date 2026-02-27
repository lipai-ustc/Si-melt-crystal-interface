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

# ===== 高斯平滑 =====
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

# ===== 计算每个 slice 的标准差 =====
def compute_std_per_slice(x_atoms, values, x_grid, dx):
    stds = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        left = x0 - dx / 2
        right = x0 + dx / 2
        mask = (x_atoms >= left) & (x_atoms < right)
        if np.any(mask):
            stds[i] = np.std(values[mask])
        else:
            stds[i] = np.nan
    return stds

# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0all_new_pot\111\cal_result\atomvolume.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\111\cal_result\interface_55.txt"
    output_path = r"E:\free_energy\0all_new_pot\111\density\aim-vor-density"
    os.makedirs(output_path, exist_ok=True)

    frames = load_atomvolume_xyz(atomvolume_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    rel_range = 20
    dx = 0.1
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    sigmas = [0.8, 0.08]

    # ===== 对齐并合并所有帧的原子 =====
    all_x = []
    all_density = []
    for (x_atoms, volumes), interface_x in zip(frames, interfaces):
        rel_x_atoms = x_atoms - interface_x
        densities = np.where(volumes != 0, 1.0 / volumes, 0.0)
        all_x.append(rel_x_atoms)
        all_density.append(densities)

    all_x = np.concatenate(all_x)
    all_density = np.concatenate(all_density)

    # ===== 对合并后的数据进行高斯平滑 =====
    avg_density_dict = {}
    std_density_dict = {}
    for sigma in sigmas:
        avg_density_dict[sigma] = gaussian_smooth(all_x, all_density, rel_grid, sigma)
        std_density_dict[sigma] = compute_std_per_slice(all_x, all_density, rel_grid, dx)

        # 保存 CSV
        np.savetxt(
            os.path.join(output_path, f"density_avg_std_sigma{sigma}.csv"),
            np.column_stack((rel_grid, avg_density_dict[sigma], std_density_dict[sigma])),
            delimiter=',',
            header="Relative_x,Avg_Density,Std_Density",
            comments=''
        )

    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(8, 5))
    fs = 23
    plot_x_min = -20
    plot_x_max = 20
    mask = (rel_grid >= plot_x_min) & (rel_grid <= plot_x_max)
    rel_grid_plot = rel_grid[mask]

    line_color = 'C0'
    alphas = {0.8:0.9, 0.08:0.2}

    for sigma in sigmas:
        avg_plot = avg_density_dict[sigma][mask]
        std_plot = std_density_dict[sigma][mask]
        ax.plot(rel_grid_plot, avg_plot, color=line_color, alpha=alphas[sigma], label=f"σ={sigma} Å")
        ax.fill_between(rel_grid_plot, avg_plot - std_plot, avg_plot + std_plot, color=line_color, alpha=0.2)

    ax.axvline(0, color='r', linestyle='--')
    ax.axvline(-10, color='r', linestyle='--')

    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Number density (Å⁻³)", fontsize=fs)
    ax.set_ylim(0.049, 0.058)
    from matplotlib.pyplot import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.002))

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.grid(False)
    ax.legend(fontsize=fs-4)
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "all_frames_density_avg_std.png"), dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"处理完成，结果保存在 {output_path}")

if __name__ == "__main__":
    main()

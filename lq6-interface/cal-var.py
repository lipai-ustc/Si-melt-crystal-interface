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


# ====== Step 4: 可视化函数（含阴影区间） ======
def plot_avg_lq6_with_variance(rel_grid, mean_values, var_values, out_path, x_min, x_max):
    std_values = np.sqrt(var_values)  # 计算标准差
    plt.figure(figsize=(8, 5))

    # 阴影区间：mean ± std
    plt.fill_between(rel_grid,
                     mean_values - std_values,
                     mean_values + std_values,
                     color='C1', alpha=0.3, label="±1σ 区间")

    # 平均值曲线
    plt.plot(rel_grid, mean_values, color='C1', label="Mean lq6")

    # 界面位置
    plt.axvline(0, color='r', linestyle='--', label="Interface")

    plt.xlabel("Relative x (Å)", fontsize=16)
    plt.ylabel("lq6", fontsize=16)
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 平均 lq6 + 方差阴影图已保存: {out_path}")


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

    smoothed_all_frames = []

    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x  # 对齐到界面
        lq6_values = frame[:, 3]
        smoothed = gaussian_smooth_lq6(x_atoms, lq6_values, rel_grid, sigma)
        smoothed_all_frames.append(smoothed)

    smoothed_all_frames = np.array(smoothed_all_frames)
    avg_smoothed = np.nanmean(smoothed_all_frames, axis=0)
    var_smoothed = np.nanvar(smoothed_all_frames, axis=0)

    # === 保存结果 ===
    output_csv = os.path.join(output_dir, "avg_var_smoothed_lq6_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed, var_smoothed]),
               delimiter=',', header='Relative_x,Mean_Smoothed_lq6,Variance', comments='')
    print(f"✅ 平均和方差已保存: {output_csv}")

    # === 绘图 ===
    plot_file = os.path.join(output_dir, "avg_smoothed_lq6_with_variance.png")
    plot_avg_lq6_with_variance(rel_grid, avg_smoothed, var_smoothed, plot_file, -rel_range, rel_range)


if __name__ == "__main__":
    main()

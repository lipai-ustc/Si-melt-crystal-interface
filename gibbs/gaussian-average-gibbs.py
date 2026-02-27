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
            smoothed[i] = np.nan  # 如果没有权重，写 nan
    return smoothed


# ===== 主程序 =====
def main():
    atomvolume_file = r"E:\free_energy\0_all_newpot-2\110\gibbs\gibbs_free_energy.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\110\cal_result\interface_55.txt"
    output_path = r"E:\free_energy\0_all_newpot-2\110\gibbs"
    os.makedirs(output_path, exist_ok=True)

    frames = load_atomvolume_xyz(atomvolume_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    # 设置相对界面的位置网格
    rel_range = 20  # Å
    rel_grid = np.linspace(-rel_range, rel_range, 400)

    sigma = 0.5 # 高斯展宽，Å

    all_density = []

    for (x_atoms, volumes), interface_x in zip(frames, interfaces):
        # 相对界面的位置
        rel_x_atoms = x_atoms - interface_x

        # 高斯平滑
        density = gaussian_smooth(rel_x_atoms, volumes, rel_grid, sigma)

        all_density.append(density)

    # 平均所有帧
    avg_density = np.mean(all_density, axis=0)

    # 保存数据
    np.savetxt(os.path.join(output_path, "volume_relative_to_interface.txt"),
               np.column_stack((rel_grid, avg_density)),
               header="Relative_x(A)    Smoothed AtomicVolume", fmt="%.5f")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_density, label="sigma0.5-100-all")
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Distance from Interface (Å)")
    plt.ylabel("Smoothed Atomic Volume")
    plt.title(f"Atomic Volume relative to Interface (Gaussian smoothing σ={sigma} Å)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"all_frames_sigma{sigma}_100.png"), dpi=300)
    plt.close()

    print(f"处理完成，结果保存在 {output_path}")

if __name__ == "__main__":
    main()

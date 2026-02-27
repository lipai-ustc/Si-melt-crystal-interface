import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ===== 读取xyz文件，只提取x坐标 =====
def load_xyz_for_density(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            _ = f.readline()  # 跳过注释行
            atoms = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                x = float(parts[1])
                atoms.append(x)
            frames.append(np.array(atoms))
    return frames

# ===== 读取interface文件 =====
def load_interfaces(filename):
    data = np.loadtxt(filename)
    return data[:, 0]  # 只取第一列，界面x位置

# ===== 高斯平滑计算密度 =====
def gaussian_density(x_atoms, x_grid, sigma):
    density = np.zeros_like(x_grid)
    for x in x_atoms:
        density += np.exp(-((x_grid - x) ** 2) / (2 * sigma ** 2))
    prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma)  # 正规化，保证总量
    return density * prefactor

# ===== 主程序 =====
def main():
    xyz_file = r"E:\free_energy\all_new_pot\100\cal-result\merge.xyz"
    interface_file = r"E:\free_energy\all_new_pot\100\cal-result\interface_lq6.txt"
    output_path = r"E:\free_energy\all_new_pot\100\density\aim-vor-density"
    os.makedirs(output_path, exist_ok=True)

    frames = load_xyz_for_density(xyz_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与界面位置数量不一致"

    # 设置相对界面的位置网格
    rel_range = 30.0  # Å
    rel_grid = np.linspace(-rel_range, rel_range, 400)  # 比如-10Å到+10Å，共200点

    sigma = 0.1  # 高斯展宽宽度，Å

    all_density = []

    for x_atoms, interface_x in zip(frames, interfaces):
        # 相对于界面的位置
        rel_x_atoms = x_atoms - interface_x

        # 在原子相对位置上做高斯平滑
        density = gaussian_density(rel_x_atoms, rel_grid, sigma)

        all_density.append(density)

    # 平均所有帧
    avg_density = np.mean(all_density, axis=0)

    # 保存数据
    np.savetxt(os.path.join(output_path, "gaussian_density_relative_to_interface.txt"), np.column_stack((rel_grid, avg_density)), header="Relative_x(A) Density(atom/A)", fmt="%.5f")

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(rel_grid, avg_density)
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Distance from Interface (Å)")
    plt.ylabel("Atomic Density (atoms/Å)")
    plt.title(f"Atomic Density relative to Interface (Gaussian smoothing σ={sigma} Å)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "all_frames_sigma0.5_100.png"), dpi=300)
    plt.close()

    print(f"处理完成，结果保存在 {output_path}")

if __name__ == "__main__":
    main()

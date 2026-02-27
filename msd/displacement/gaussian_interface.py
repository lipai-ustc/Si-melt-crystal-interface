import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian_smooth_displacement(rel_x_atoms, displacements, rel_grid, sigma):
    smoothed = np.zeros_like(rel_grid)
    for i, x0 in enumerate(rel_grid):
        weights = np.exp(-((rel_x_atoms - x0) ** 2) / (2 * sigma ** 2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * displacements) / np.sum(weights)
        else:
            smoothed[i] = np.nan
    return smoothed

def read_all_frames_displacement_xyz(filepath):
    """读取所有帧，返回列表，每个元素是 N x 4 的 numpy 数组 (x, y, z, displacement)"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header = lines[i + 1]
        atoms = lines[i + 2: i + 2 + num_atoms]
        data = []
        for line in atoms:
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            displacement = float(parts[4])
            data.append([x, y, z, displacement])
        frames.append(np.array(data))
        i += 2 + num_atoms
    return frames

def load_interfaces(interface_file):
    """读取 interface.txt，返回界面位置数组"""
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

def main():
    # ==== 参数 ====
    input_file = r"E:\free_energy\all_new_pot\110\msd\msd_output.xyz"
    interface_file = r"E:\free_energy\all_new_pot\110\cal-result\interface_lq6.txt"
    output_dir = r"E:\free_energy\all_new_pot\110\msd"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.5
    dx = 0.1
    rel_range = 40

    # ==== 读取所有帧 ====
    frames = read_all_frames_displacement_xyz(input_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "❌ 接口位置数和帧数不一致！"

    # 统一相对界面坐标网格
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    smoothed_all_frames = []

    for frame, interface_x in zip(frames, interfaces):
        rel_x_atoms = frame[:, 0] - interface_x
        displacements = frame[:, 3]
        smoothed = gaussian_smooth_displacement(rel_x_atoms, displacements, rel_grid, sigma)
        smoothed_all_frames.append(smoothed)

    smoothed_all_frames = np.array(smoothed_all_frames)
    avg_smoothed = np.nanmean(smoothed_all_frames, axis=0)

    # === 保存 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_displacement_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed]),
               delimiter=',', header='Relative_x,Smoothed_Displacement', comments='')
    print(f"✅ 平均位移分布已保存：{output_csv}")

    # === 可视化 ===
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_smoothed, color='C0', label=f"σ={sigma}")
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Relative x (Å)")
    plt.ylabel("Smoothed Displacement")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "avg_smoothed_displacement_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"✅ 图像已保存：{plot_file}")

if __name__ == "__main__":
    main()

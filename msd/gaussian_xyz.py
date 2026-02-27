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
    """
    读取所有帧，返回列表，每个元素是 N x 6 的 numpy 数组:
    columns = (x, y, z, disp_x2, disp_y2, disp_z2)
    """
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
            disp_x2 = float(parts[4])
            disp_y2 = float(parts[5])
            disp_z2 = float(parts[6])
            data.append([x, y, z, disp_x2, disp_y2, disp_z2])
        frames.append(np.array(data))
        i += 2 + num_atoms
    return frames

def load_interfaces(interface_file):
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

def main():
    # ==== 参数 ====
    input_file = r"E:\free_energy\0all_new_pot\100\msd\msd2_output.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\100\cal_result\interface_lq6.txt"
    output_dir = r"E:\free_energy\0all_new_pot\100\msd"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.5
    dx = 0.1
    rel_range = 40

    # ==== 读取所有帧 ====
    frames = read_all_frames_displacement_xyz(input_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "❌ 接口位置数和帧数不一致！"

    # 相对界面位置网格
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    # 存储所有帧的平滑结果
    smoothed_x2_all = []
    smoothed_y2_all = []
    smoothed_z2_all = []

    for frame, interface_x in zip(frames, interfaces):
        rel_x_atoms = frame[:, 0] - interface_x
        disp_x2 = frame[:, 3]
        disp_y2 = frame[:, 4]
        disp_z2 = frame[:, 5]

        smoothed_x2 = gaussian_smooth_displacement(rel_x_atoms, disp_x2, rel_grid, sigma)
        smoothed_y2 = gaussian_smooth_displacement(rel_x_atoms, disp_y2, rel_grid, sigma)
        smoothed_z2 = gaussian_smooth_displacement(rel_x_atoms, disp_z2, rel_grid, sigma)

        smoothed_x2_all.append(smoothed_x2)
        smoothed_y2_all.append(smoothed_y2)
        smoothed_z2_all.append(smoothed_z2)

    smoothed_x2_all = np.array(smoothed_x2_all)
    smoothed_y2_all = np.array(smoothed_y2_all)
    smoothed_z2_all = np.array(smoothed_z2_all)

    avg_smoothed_x2 = np.nanmean(smoothed_x2_all, axis=0)
    avg_smoothed_y2 = np.nanmean(smoothed_y2_all, axis=0)
    avg_smoothed_z2 = np.nanmean(smoothed_z2_all, axis=0)

    # === 保存 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_displacement_xyz2_relative_to_interface.csv")
    np.savetxt(output_csv,
               np.column_stack([rel_grid, avg_smoothed_x2, avg_smoothed_y2, avg_smoothed_z2]),
               delimiter=',',
               header='Relative_x,Smoothed_disp_x2,Smoothed_disp_y2,Smoothed_disp_z2',
               comments='')
    print(f"✅ 平均位移平方分布已保存：{output_csv}")

    # === 可视化 ===
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_smoothed_x2, label='disp_x2', color='C0')
    plt.plot(rel_grid, avg_smoothed_y2, label='disp_y2', color='C1')
    plt.plot(rel_grid, avg_smoothed_z2, label='disp_z2', color='C2')
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Relative x (Å)")
    plt.ylabel("Smoothed Displacement^2")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "avg_smoothed_displacement_xyz2_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"✅ 图像已保存：{plot_file}")

if __name__ == "__main__":
    main()

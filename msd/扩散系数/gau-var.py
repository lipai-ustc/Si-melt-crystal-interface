import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian_smooth(x_atoms, values, x_grid, sigma):
    """对合并后的原子在网格上高斯平滑"""
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        w_sum = np.sum(weights)
        if w_sum > 0:
            smoothed[i] = np.sum(weights * values) / w_sum
        else:
            smoothed[i] = np.nan
    return smoothed

def read_all_frames_displacement_xyz(filepath):
    """读取所有帧，返回列表，每个元素是 N x 6 的 numpy 数组"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        _ = lines[i + 1]
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
    input_file = r"E:\free_energy\all_new_pot\111\msd\diffusion\diffusion.xyz"
    interface_file = r"E:\free_energy\all_new_pot\111\cal-result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\111\msd"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 1.0
    dx = 0.1
    rel_range = 20
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    # ==== 读取并对齐所有帧 ====
    frames = read_all_frames_displacement_xyz(input_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "❌ 接口位置数和帧数不一致！"

    all_rel_x = []
    all_disp_x2 = []
    all_disp_y2 = []
    all_disp_z2 = []

    for frame, interface_x in zip(frames, interfaces):
        rel_x = frame[:, 0] - interface_x
        all_rel_x.append(rel_x)
        all_disp_x2.append(frame[:, 3])
        all_disp_y2.append(frame[:, 4])
        all_disp_z2.append(frame[:, 5])

    all_rel_x = np.concatenate(all_rel_x)
    all_disp_x2 = np.concatenate(all_disp_x2)
    all_disp_y2 = np.concatenate(all_disp_y2)
    all_disp_z2 = np.concatenate(all_disp_z2)

    # === 平滑 ===
    avg_smoothed_x2 = gaussian_smooth(all_rel_x, all_disp_x2, rel_grid, sigma)
    avg_smoothed_y2 = gaussian_smooth(all_rel_x, all_disp_y2, rel_grid, sigma)
    avg_smoothed_z2 = gaussian_smooth(all_rel_x, all_disp_z2, rel_grid, sigma)

    # === 每个 slice 计算标准差 ===
    slice_std_x2 = []
    slice_std_y2 = []
    slice_std_z2 = []
    for x0 in rel_grid:
        mask = (all_rel_x >= x0 - dx / 2) & (all_rel_x < x0 + dx / 2)
        slice_std_x2.append(np.std(all_disp_x2[mask]) if np.any(mask) else np.nan)
        slice_std_y2.append(np.std(all_disp_y2[mask]) if np.any(mask) else np.nan)
        slice_std_z2.append(np.std(all_disp_z2[mask]) if np.any(mask) else np.nan)
    slice_std_x2 = np.array(slice_std_x2)
    slice_std_y2 = np.array(slice_std_y2)
    slice_std_z2 = np.array(slice_std_z2)

    # === 保存 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_displacement_xyz2_with_std.csv")
    np.savetxt(output_csv,
               np.column_stack([rel_grid, avg_smoothed_x2, slice_std_x2,
                                avg_smoothed_y2, slice_std_y2,
                                avg_smoothed_z2, slice_std_z2]),
               delimiter=',',
               header='Relative_x,Avg_disp_x2,Std_disp_x2,Avg_disp_y2,Std_disp_y2,Avg_disp_z2,Std_disp_z2',
               comments='')
    print(f"✅ 平均+标准差结果保存：{output_csv}")

    # === 可视化（以 x2 为例）===
    plt.figure(figsize=(8, 5))
    plt.fill_between(rel_grid,
                     avg_smoothed_x2 - slice_std_x2,
                     avg_smoothed_x2 + slice_std_x2,
                     color='C0', alpha=0.3, linewidth=0)
    plt.plot(rel_grid, avg_smoothed_x2, label='disp_x2 ± σ', color='C0')
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Relative x (Å)")
    plt.ylabel("Displacement²")
    plt.legend(frameon=False)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "avg_smoothed_displacement_x2_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"✅ 图像已保存：{plot_file}")

if __name__ == "__main__":
    main()

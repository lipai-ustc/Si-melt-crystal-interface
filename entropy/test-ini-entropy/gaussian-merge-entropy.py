import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian_smooth_entropy(x_atoms, entropies, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * entropies) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

def read_all_frames_merge_xyz(filepath):
    """读取 merge.xyz 文件，返回列表，每个元素是 N x 5 的 numpy 数组 (x, y, z, entropy, frame_index)"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    frame_id = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header = lines[i + 1]
        atoms = lines[i + 2: i + 2 + num_atoms]
        data = []
        for line in atoms:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = map(float, parts[1:4])
            entropy = float(parts[5])  # 注意索引：第6列为 entropy
            data.append([x, y, z, entropy])
        frames.append(np.array(data))
        i += 2 + num_atoms
        frame_id += 1
    return frames

def load_interfaces(interface_file):
    """读取 interface.txt，返回界面位置数组"""
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

def main():
    # === 参数设置 ===
    merge_xyz_file = r"E:\free_energy\0all_new_pot\111\cal_result\merge.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\111\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\111\entropy\S2"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    dx = 0.1
    rel_range = 40

    # === 读取所有帧和界面位置 ===
    frames = read_all_frames_merge_xyz(merge_xyz_file)
    interfaces = load_interfaces(interface_file)
    assert len(frames) == len(interfaces), "帧数和 interface.txt 不一致"

    rel_grid = np.arange(-rel_range, rel_range + dx, dx)
    smoothed_all_frames = []

    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x
        entropies = frame[:, 3]
        smoothed = gaussian_smooth_entropy(x_atoms, entropies, rel_grid, sigma)
        smoothed_all_frames.append(smoothed)

    smoothed_all_frames = np.array(smoothed_all_frames)
    avg_smoothed = np.nanmean(smoothed_all_frames, axis=0)

    # === 保存 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_entropy_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed]),
               delimiter=',', header='Relative_x,Smoothed_Entropy', comments='')
    print(f"✅ 平均熵分布保存：{output_csv}")

    # === 可视化 ===
    plt.figure(figsize=(8, 5))
    plt.plot(rel_grid, avg_smoothed, color='C1', label=f"σ={sigma}")
    plt.axvline(0, color='r', linestyle='--', label='Interface')
    plt.xlabel("Relative x (Å)")
    plt.ylabel("Smoothed Entropy (-TS, eV/atom)")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "avg_smoothed_entropy_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"✅ 图像保存：{plot_file}")

if __name__ == "__main__":
    main()

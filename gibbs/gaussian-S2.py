import numpy as np
import matplotlib.pyplot as plt

def gaussian_smooth_entropy(frame, x_grid, sigma):
    x_coords = frame[:, 0]         # x 坐标
    entropies = frame[:, 3]        # entropy（第 4 列，因 frame 为 [x, y, z, entropy]）
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_coords - x0) ** 2) / (2 * sigma ** 2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * entropies) / np.sum(weights)
        else:
            smoothed[i] = np.nan
    return smoothed

def read_merge_xyz_entropy_frame(filepath, target_frame=1):
    """读取 merge.xyz 中指定帧的数据，返回 numpy 数组 [x, y, z, entropy]"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    current_frame = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header = lines[i + 1]
        if "Frame=" in header:
            frame_num = int(header.split("Frame=")[1].strip())
        else:
            current_frame += 1
            frame_num = current_frame

        if frame_num == target_frame:
            atoms = lines[i + 2: i + 2 + num_atoms]
            data = []
            for line in atoms:
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                entropy = float(parts[5])  # 第六列是 entropy
                data.append([x, y, z, entropy])
            return np.array(data)

        i += 2 + num_atoms

    raise ValueError(f"未找到指定的 Frame={target_frame}")

def main():
    input_file = r"E:\free_energy\all_new_pot\s100\cal\merge.xyz"
    output_csv = "smoothed_entropyS2.csv"
    frame_to_use = 11     # <<< 修改为你要分析的帧号
    sigma = 0.5           # 高斯核宽度
    dx = 0.5              # 网格间距

    frame = read_merge_xyz_entropy_frame(input_file, target_frame=frame_to_use)

    # 自动确定 x 范围
    x_vals = frame[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    x_grid = np.arange(x_min, x_max + dx, dx)

    smoothed = gaussian_smooth_entropy(frame, x_grid, sigma)

    # 保存为 CSV
    np.savetxt(output_csv, np.column_stack([x_grid, smoothed]),
               delimiter=',', header='x,smoothed_entropy', comments='')

    print(f"✅ Frame={frame_to_use} 熵高斯平滑完成，结果保存为 {output_csv}")

    # 可视化
    plt.plot(x_grid, smoothed, label=f"Frame={frame_to_use}, σ={sigma}")
    plt.xlabel("x")
    plt.ylabel("Smoothed Entropy")
    plt.title(f"Entropy Smoothed Along X (Frame {frame_to_use})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("smoothed_entropyS2.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

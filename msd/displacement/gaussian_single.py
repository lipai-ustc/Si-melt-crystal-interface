import numpy as np
import matplotlib.pyplot as plt

def gaussian_smooth_displacement(frame, x_grid, sigma):
    x_coords = frame[:, 0]              # x 坐标
    displacements = frame[:, 3]         # displacement（第 4 列）
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_coords - x0) ** 2) / (2 * sigma ** 2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * displacements) / np.sum(weights)
        else:
            smoothed[i] = np.nan
    return smoothed

def read_xyz_displacement_frame(filepath, target_frame=11):
    """读取包含 displacement 的 xyz 文件中的指定帧，返回 [x, y, z, displacement]"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    current_frame = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header = lines[i + 1]
        if "Frame=" in header:
            frame_num = int(header.split("Frame=")[1].strip().split()[0])
        else:
            current_frame += 1
            frame_num = current_frame

        if frame_num == target_frame:
            atoms = lines[i + 2: i + 2 + num_atoms]
            data = []
            for line in atoms:
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                displacement = float(parts[4])  # 第五列是 displacement
                data.append([x, y, z, displacement])
            return np.array(data)

        i += 2 + num_atoms

    raise ValueError(f"未找到指定的 Frame={target_frame}")

def main():
    input_file = r"E:\free_energy\all_new_pot\100\msd\msd_output.xyz"   # <<< 替换为你的实际文件路径
    output_csv = "smoothed_displacement.csv"
    frame_to_use =100
    sigma = 0.5
    dx = 0.1

    frame = read_xyz_displacement_frame(input_file, target_frame=frame_to_use)

    x_vals = frame[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    x_grid = np.arange(x_min, x_max + dx, dx)

    smoothed = gaussian_smooth_displacement(frame, x_grid, sigma)

    np.savetxt(output_csv, np.column_stack([x_grid, smoothed]),
               delimiter=',', header='x,smoothed_displacement', comments='')

    print(f"✅ Frame={frame_to_use} 位移高斯平滑完成，结果保存为 {output_csv}")

    plt.plot(x_grid, smoothed, label=f"Frame={frame_to_use}, σ={sigma}")
    plt.xlabel("x")
    plt.ylabel("Smoothed Displacement")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("smoothed_displacement_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

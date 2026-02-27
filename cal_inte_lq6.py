import numpy as np

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

# ====== Step 2: 高斯平滑 lq6 函数 ======
def gaussian_smooth_lq6(frame, x_grid, sigma):
    x_coords = frame[:, 0]
    lq6_values = frame[:, 3]
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_coords - x0) ** 2) / (2 * sigma ** 2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * lq6_values) / np.sum(weights)
        else:
            smoothed[i] = np.nan
    return smoothed

# ====== Step 3: 找到目标 lq6 对应的界面位置 ======
def find_interface_x(smoothed_lq6, x_grid, target_lq6=0.55):
    diff = np.abs(smoothed_lq6 - target_lq6)
    index = np.nanargmin(diff)
    return x_grid[index]

# ====== 主函数：提取界面位置并输出 ======
def main():
    filename = r"E:\free_energy\all_new_pot\111\cal-result\merge.xyz"  # 输入文件路径
    output_file = r"E:\free_energy\all_new_pot\111\cal-result\interface_55.txt"  # 输出文件路径

    frames = load_xyz_with_lq6(filename)

    x_min, x_max = 60, 100
    x_bins = 800
    x_grid = np.linspace(x_min, x_max, x_bins)
    sigma = 0.5
    target_lq6 = 0.55

    interface_list = []

    for frame in frames:
        smoothed = gaussian_smooth_lq6(frame, x_grid, sigma)
        interface_x = find_interface_x(smoothed, x_grid, target_lq6)
        lq6_at_interface = np.interp(interface_x, x_grid, smoothed)
        interface_list.append((interface_x, lq6_at_interface))

    np.savetxt(output_file,
               np.array(interface_list),
               header="x_position lq6", fmt="%.5f")

if __name__ == "__main__":
    main()

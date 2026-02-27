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
def find_interface_x(smoothed_lq6, x_grid, target_lq6=0.6):
    diff = np.abs(smoothed_lq6 - target_lq6)
    index = np.nanargmin(diff)
    return x_grid[index]

# ====== 主函数：提取界面位置并输出 ======
def main():
    filename = r"E:\free_energy\all_new_pot\100\cal-result\merge.xyz"  # 输入文件路径
    output_file = r"E:\free_energy\all_new_pot\100\cal-result\interface_lq6-bins13.txt"  # 输出文件路径

    frames = load_xyz_with_lq6(filename)

    x_min, x_max = 60, 100
    x_bins = 800  # finer grid for smoother interpolation
    x_grid = np.linspace(x_min, x_max, x_bins)
    sigma = 0.5
    target_lq6 = 0.6

    interface_list = []

    # ---- 第一步：用第一帧找出基准界面位置 ----
    first_frame = frames[0]
    smoothed_first = gaussian_smooth_lq6(first_frame, x_grid, sigma)
    initial_interface_x = find_interface_x(smoothed_first, x_grid, target_lq6)
    lq6_at_initial = np.interp(initial_interface_x, x_grid, smoothed_first)
    interface_list.append((initial_interface_x, lq6_at_initial))

    print(f"First frame interface at x = {initial_interface_x:.5f}")

    # ---- 第二步：后续帧在±3.1×(1..6)附近找最接近0.6的位置 ----
    delta = 1.35
    multipliers = np.arange(-6, 7)  # -6 to +6 inclusive
    candidate_offsets = delta * multipliers

    for frame in frames[1:]:
        smoothed = gaussian_smooth_lq6(frame, x_grid, sigma)

        # 13个采样位置
        candidate_positions = initial_interface_x + candidate_offsets
        candidate_positions = np.clip(candidate_positions, x_min, x_max)  # keep in grid bounds
        candidate_lq6 = np.interp(candidate_positions, x_grid, smoothed)

        # 选择最接近0.6的位置
        diff = np.abs(candidate_lq6 - target_lq6)
        best_index = np.argmin(diff)
        best_x = candidate_positions[best_index]
        best_lq6 = candidate_lq6[best_index]

        interface_list.append((best_x, best_lq6))

    # ---- 保存到文件 ----
    np.savetxt(output_file,
               np.array(interface_list),
               header="x_position lq6", fmt="%.5f")

if __name__ == "__main__":
    main()

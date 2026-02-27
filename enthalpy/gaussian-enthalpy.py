import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian_smooth_energy(x_atoms, energies, x_grid, sigma):
    smoothed = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        weights = np.exp(-((x_atoms - x0) ** 2) / (2 * sigma ** 2))
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            smoothed[i] = np.sum(weights * energies) / weight_sum
        else:
            smoothed[i] = np.nan
    return smoothed

def read_all_frames_merge_xyz(filepath):
    """读取所有帧，返回列表，每个元素是 N x 4 的 numpy 数组 (x,y,z,energy)"""
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
            energy = float(parts[6])
            data.append([x, y, z, energy])
        frames.append(np.array(data))
        i += 2 + num_atoms
    return frames

def load_interfaces(interface_file):
    """读取 interface.txt，返回界面位置数组"""
    data = np.loadtxt(interface_file)
    return data[:, 0] if data.ndim > 1 else data

def plot_and_save_smoothed_energy(
        centers, energy, out_path, sigma,
        xlabel="z (Å)", ylabel="Enthalpy (eV/atom)",
        x_min_plot=None, x_max_plot=None
    ):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    #ax.set_title(title, fontsize=fs)

    # 可选截取 x 范围
    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(centers, dtype=bool)
        if x_min_plot is not None:
            mask &= (centers >= x_min_plot)
        if x_max_plot is not None:
            mask &= (centers <= x_max_plot)
        centers = centers[mask]
        energy = energy[mask]

    ax.plot(centers, energy, color='C0', alpha=0.9)
    ax.axvline(0, color='r', linestyle='--')

    from matplotlib.pyplot import MultipleLocator
    y_major_locator = MultipleLocator(0.12)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    ax.set_ylim(-9.88, -9.28)


    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")

def main():
    # === 参数设置 ===
    merge_xyz_file = r"E:\free_energy\0all_new_pot\110\cal_result\merge.xyz"
    interface_file = r"E:\free_energy\0all_new_pot\110\cal_result\interface_55.txt"
    output_dir = r"E:\free_energy\0_all_newpot-2\110\enthalpy"
    os.makedirs(output_dir, exist_ok=True)

    sigma = 0.8
    dx = 0.1
    rel_range = 20

    # === 读取所有帧和对应界面位置 ===
    frames = read_all_frames_merge_xyz(merge_xyz_file)
    interfaces = load_interfaces(interface_file)

    assert len(frames) == len(interfaces), "帧数与 interface.txt 不一致"

    # === 定义统一对齐网格 ===
    rel_grid = np.arange(-rel_range, rel_range + dx, dx)

    smoothed_all_frames = []

    for frame, interface_x in zip(frames, interfaces):
        x_atoms = frame[:, 0] - interface_x       # 对齐到界面
        energies = frame[:, 3]

        smoothed = gaussian_smooth_energy(x_atoms, energies, rel_grid, sigma)
        smoothed_all_frames.append(smoothed)

    smoothed_all_frames = np.array(smoothed_all_frames)
    avg_smoothed = np.nanmean(smoothed_all_frames, axis=0)

    # === 保存为 CSV ===
    output_csv = os.path.join(output_dir, "avg_smoothed_energy_relative_to_interface.csv")
    np.savetxt(output_csv, np.column_stack([rel_grid, avg_smoothed]),
               delimiter=',', header='Relative_x,Smoothed_Energy', comments='')

    print(f"✅ 所有帧处理完成，结果保存：{output_csv}")

    # === 可视化（新风格）===
    plot_file = os.path.join(output_dir, "avg_smoothed_energy_plot.png")
    plot_and_save_smoothed_energy(
        rel_grid, avg_smoothed,
        out_path=plot_file,
        sigma=sigma,
        x_min_plot=-rel_range,
        x_max_plot=rel_range
    )

if __name__ == "__main__":
    main()

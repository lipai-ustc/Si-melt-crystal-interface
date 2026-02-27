import numpy as np
import matplotlib.pyplot as plt
import os
import re

######################################
# 用户输入参数
######################################

xyz_file = r"E:\free_energy\all_new_pot\100\cal-result\merge.xyz"
interface_file = r"E:\free_energy\all_new_pot\100\cal-result\interface_lq6-bins13.txt"
output_dir = r"E:\free_energy\all_new_pot\100\density\test-lq6"
output_image_prefix = "density_frame"

box_length_y = 52.8131
box_length_z = 56.0168
bin_width = 0.05

plot_x_min = -20
plot_x_max = 20

# 【修改 1】这里用列表指定你想要的帧
#selected_frames = [0-20]
selected_frames = list(range(0, 21))
######################################
def load_interface_positions(filename):
    positions = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            if line.strip():
                parts = line.strip().split()
                positions.append(float(parts[0]))
    return np.array(positions)

######################################
def load_selected_xyz_frames(filename, selected_frames, interface_positions):
    """
    加载xyz文件里指定帧号的所有原子x - interface位移后的列表
    返回: dict{frame_number: shifted_x_array}
    """
    selected_set = set(selected_frames)
    frames_data = dict()

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                num_atoms = int(line.strip())
            except ValueError:
                print(f"[WARN] Unexpected line (not integer atom count): {line}")
                continue

            comment_line = f.readline()
            frame_match = re.search(r'Frame\s*=\s*(\d+)', comment_line)
            if frame_match:
                current_frame_number = int(frame_match.group(1))
            else:
                print(f"[WARN] No Frame info in comment line: {comment_line.strip()}")
                for _ in range(num_atoms):
                    f.readline()
                continue

            if current_frame_number not in selected_set:
                # skip this frame
                for _ in range(num_atoms):
                    f.readline()
                continue

            if current_frame_number >= len(interface_positions):
                print(f"[WARN] Frame number {current_frame_number} exceeds interface data size")
                for _ in range(num_atoms):
                    f.readline()
                continue

            interface_pos = interface_positions[current_frame_number]
            atoms = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                x = float(parts[1])
                shifted_x = x - interface_pos
                atoms.append(shifted_x)

            frames_data[current_frame_number] = np.array(atoms)

    print(f"[INFO] Loaded frames: {sorted(frames_data.keys())}")
    return frames_data

######################################
def compute_density_single_frame(atoms_x, bin_width, box_length_y, box_length_z):
    bins = np.arange(plot_x_min, plot_x_max + bin_width, bin_width)
    counts, edges = np.histogram(atoms_x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_volume = bin_width * box_length_y * box_length_z
    number_density = counts / bin_volume
    return centers, number_density

######################################
def plot_and_save_density(centers, density, title, xlabel, ylabel, out_path,
                           x_min_plot=None, x_max_plot=None):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)

    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(centers, dtype=bool)
        if x_min_plot is not None:
            mask &= (centers >= x_min_plot)
        if x_max_plot is not None:
            mask &= (centers <= x_max_plot)
        centers = centers[mask]
        density = density[mask]

    ax.plot(centers, density, color='C0', alpha=0.9)
    ax.axvline(0, color='r', linestyle='--')
    ax.axvline(-10, color='r', linestyle='--')

    from matplotlib.pyplot import MultipleLocator
    y_major_locator = MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    if x_min_plot is not None and x_max_plot is not None:
        ax.set_xlim(x_min_plot, x_max_plot)

    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")

######################################
if __name__ == "__main__":
    print("[START] Loading interface positions...")
    interface_positions = load_interface_positions(interface_file)
    print(f"[INFO] Loaded interface positions: {len(interface_positions)}")

    print("[START] Loading xyz file...")
    frames_data = load_selected_xyz_frames(
        xyz_file,
        selected_frames,
        interface_positions
    )

    if not frames_data:
        print("[ERROR] No frames found in specified list!")
        exit()

    os.makedirs(output_dir, exist_ok=True)

    for frame_number in sorted(frames_data.keys()):
        print(f"[START] Computing density for frame {frame_number}...")
        centers, density = compute_density_single_frame(
            frames_data[frame_number],
            bin_width=bin_width,
            box_length_y=box_length_y,
            box_length_z=box_length_z
        )

        out_image_path = os.path.join(
            output_dir,
            f"{output_image_prefix}_{frame_number}.png"
        )

        plot_and_save_density(
            centers,
            density,
            title=f'Density profile (Frame {frame_number})',
            xlabel='z (Å)',
            ylabel='Number density (Å⁻³)',
            out_path=out_image_path,
            x_min_plot=plot_x_min,
            x_max_plot=plot_x_max
        )

    print("[DONE] All frames processed!")

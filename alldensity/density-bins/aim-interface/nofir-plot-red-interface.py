import numpy as np
import matplotlib.pyplot as plt
import os
import re

######################################
# 用户输入参数
######################################

xyz_file = r"E:\free_energy\0dispot\100\result\merge.xyz"
interface_file = r"E:\free_energy\0dispot\100\result\interface_55-bins.txt"
output_dir = r"E:\free_energy\0dispot\100\density\bins"
output_image_name_fine = "biglq6-try55-3.png"
output_data_name_fine = "biglq6_density_data-3.txt"

box_length_y = 53.6126#100
box_length_z = 53.6126
#box_length_y =52.8256#110
#box_length_z = 56.03
#box_length_y = 60.8935#111
#box_length_z = 52.7353
bin_width = 0.05

start_frame = 0
end_frame = 400

plot_x_min = -20
plot_x_max = 20

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
def load_xyz_for_density(filename, start_frame, end_frame, interface_positions):
    frames = []
    current_frame_number = None

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
                current_frame_number = None

            if current_frame_number is None or current_frame_number < start_frame or current_frame_number > end_frame:
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

            frames.append(np.array(atoms))

    print(f"[INFO] Frames loaded in range [{start_frame}, {end_frame}]: {len(frames)}")
    return frames

######################################
def compute_density(frames, bin_width, box_length_y, box_length_z):
    all_x = np.concatenate(frames)
    print(f"[INFO] Total x-coordinates from selected frames: {len(all_x)}")

    x_min = plot_x_min
    x_max = plot_x_max
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    counts, edges = np.histogram(all_x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_volume = bin_width * box_length_y * box_length_z
    avg_counts_per_frame = counts / len(frames)
    number_density = avg_counts_per_frame / bin_volume

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
    #ax.axvline(-7.86967, color='r', linestyle='--')
    #ax.axvline( -0.85213, color='r', linestyle='--')

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
def save_density_data(centers, density, output_file):
    """
    Save density profile data to a text file.
    Columns: bin_center, number_density
    """
    header = "bin_center(A) number_density(1/A^3)"
    data = np.column_stack((centers, density))
    np.savetxt(output_file, data, header=header, fmt="%.6f")
    print(f"[SUCCESS] Density data saved to: {output_file}")

######################################
if __name__ == "__main__":
    print("[START] Loading interface positions...")
    interface_positions = load_interface_positions(interface_file)
    print(f"[INFO] Loaded interface positions: {len(interface_positions)}")

    print("[START] Loading xyz file...")
    frames = load_xyz_for_density(
        xyz_file,
        start_frame,
        end_frame,
        interface_positions
    )

    if not frames:
        print("[ERROR] No frames found in specified range!")
        exit()

    print("[START] Computing density profile...")
    centers, fine_density = compute_density(
        frames,
        bin_width=bin_width,
        box_length_y=box_length_y,
        box_length_z=box_length_z
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plot_and_save_density(
        centers,
        fine_density,
        title=f'Fine-scale density profile (Frames {start_frame}-{end_frame})',
        xlabel='z (Å)',
        ylabel='Number density (Å⁻³)',
        out_path=os.path.join(output_dir, output_image_name_fine),
        x_min_plot=plot_x_min,
        x_max_plot=plot_x_max
    )

    # Save data
    save_density_data(
        centers,
        fine_density,
        os.path.join(output_dir, output_data_name_fine)
    )

    print("[DONE] All finished!")

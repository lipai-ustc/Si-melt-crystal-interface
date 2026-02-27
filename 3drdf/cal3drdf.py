import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

##############################################
# 用户输入路径
##############################################
merge_xyz_file = r"E:\free_energy\all_new_pot\111\cal-result\merge.xyz"
interface_ABCD_file = r"E:\free_energy\all_new_pot\111\3drdf\ABCDE_positions.txt"
output_data_file = r"E:\free_energy\all_new_pot\111\3drdf\ABCDE_g3d_result.txt"
output_image_file = r"E:\free_energy\all_new_pot\111\3drdf\ABCDE_g3d_plot.png"

# RDF 参数
r_max = 10.0
dr = 0.1

##############################################
# ① 读取interface_ABCD_real_positions.txt
##############################################
def load_layer_ranges(filename):
    layer_data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    current_frame = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Frame"):
            current_frame = int(line.split()[1])
            layer_data[current_frame] = {}
        else:
            parts = line.split()
            layer = parts[0]
            x_min = float(parts[1])
            x_max = float(parts[2])
            layer_data[current_frame][layer] = (x_min, x_max)
    return layer_data

print("[INFO] Loading layer ranges...")
layer_ranges_per_frame = load_layer_ranges(interface_ABCD_file)
print(f"[INFO] Loaded layer ranges for {len(layer_ranges_per_frame)} frames.")

##############################################
# ② 读取trajectory
##############################################
print("[INFO] Loading trajectory...")
trajectory = read(merge_xyz_file + "@:", index=":")
print(f"[INFO] Loaded {len(trajectory)} frames.")

##############################################
# ③ 计算3D RDF（以中心原子为W/X/Y/Z层内原子）
##############################################
def compute_g3d(center_positions, all_positions, r_max, dr, box_xyz):
    N_centers = len(center_positions)
    if N_centers == 0:
        return None, None

    distances = []
    for center in center_positions:
        delta = all_positions - center
        for dim in range(3):
            box_length = box_xyz[dim]
            delta[:, dim] -= box_length * np.round(delta[:, dim] / box_length)
        dists = np.linalg.norm(delta, axis=1)
        in_range = (dists < r_max) & (dists > 1e-6)
        distances.extend(dists[in_range])

    distances = np.array(distances)
    bins = np.arange(0, r_max + dr, dr)
    hist, edges = np.histogram(distances, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])

    volume = box_xyz[0] * box_xyz[1] * box_xyz[2]
    density = len(all_positions) / volume
    shell_volume = 4.0 * np.pi * r**2 * dr
    ideal_counts = density * shell_volume * N_centers

    g3d = hist / ideal_counts
    return r, g3d

##############################################
# ④ 主循环：逐帧逐层计算
##############################################
all_layers = ["A", "B", "C", "D","E"]
g3d_accum = {layer: [] for layer in all_layers}

print("[INFO] Starting main loop over frames...")
for frame_idx, atoms in enumerate(trajectory):
    if frame_idx not in layer_ranges_per_frame:
        continue
    layer_ranges = layer_ranges_per_frame[frame_idx]
    x_coords = atoms.positions[:, 0]

    box_lengths = atoms.cell.lengths()  # (Lx, Ly, Lz)
    box_xyz = tuple(box_lengths)

    positions_all = atoms.positions

    for layer in all_layers:
        if layer not in layer_ranges:
            continue
        x_min, x_max = layer_ranges[layer]
        mask = (x_coords >= x_min) & (x_coords < x_max)
        center_positions = atoms.positions[mask]
        if len(center_positions) < 1:
            continue

        r, g3d = compute_g3d(center_positions, positions_all, r_max, dr, box_xyz)
        if g3d is not None:
            g3d_accum[layer].append(g3d)

print("[INFO] Finished accumulating 3D RDFs over all frames.")

##############################################
# ⑤ 对所有帧平均
##############################################
average_g3d = {}
for layer in all_layers:
    data = np.array(g3d_accum[layer])
    if len(data) == 0:
        print(f"[WARN] No data for layer {layer}")
        continue
    avg = np.mean(data, axis=0)
    average_g3d[layer] = avg

##############################################
# ⑥ 绘图
##############################################
plt.figure(figsize=(8, 6))
for layer in all_layers:
    if layer in average_g3d:
        plt.plot(r, average_g3d[layer], label=f"Layer {layer}")

plt.xlabel("r (Å)", fontsize=14)
plt.ylabel("g₃D(r)", fontsize=14)
plt.title("3D RDF for 100 ABCD layers", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Plot image saved to: {output_image_file}")
plt.close()

##############################################
# ⑦ 保存数据
##############################################
print("[INFO] Saving g₃D(r) data to file...")
with open(output_data_file, 'w') as f:
    header_line = "r"
    for layer in all_layers:
        header_line += f"\tg3D_{layer}"
    f.write(header_line + "\n")

    for i in range(len(r)):
        line = f"{r[i]:.6f}"
        for layer in all_layers:
            if layer in average_g3d:
                line += f"\t{average_g3d[layer][i]:.6f}"
            else:
                line += f"\tNaN"
        f.write(line + "\n")

print(f"[SUCCESS] Data saved to: {output_data_file}")
print("[SUCCESS] All done!")

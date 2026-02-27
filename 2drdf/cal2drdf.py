import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

##############################################
# 用户输入路径
##############################################
merge_xyz_file = r"E:\free_energy\all_new_pot\100\cal-result\merge.xyz"
interface_ABCD_file = r"E:\free_energy\all_new_pot\100\2drdf\WXYZ_positions.txt"
output_data_file = r"E:\free_energy\all_new_pot\100\2drdf\WXYZ_g2d_result.txt"
output_image_file = r"E:\free_energy\all_new_pot\100\2drdf\WXYZ_g2d_plot.png"

# RDF 参数
r_max = 20.0
dr = 0.1

##############################################
# ① 读取interface_ABCD_real_positions.txt
##############################################
def load_layer_ranges(filename):
    """
    返回: { frame_index : {layer_name: (x_min, x_max)} }
    """
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
# ③ 计算2D pair distribution function
##############################################
def compute_g2d(positions2d, r_max, dr, box_yz):
    """
    positions2d: N x 2 array of (y, z) positions
    """
    N = len(positions2d)
    if N < 2:
        return None, None

    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            dy = positions2d[i,0] - positions2d[j,0]
            dz = positions2d[i,1] - positions2d[j,1]
            # PBC in y and z
            dy -= box_yz[0] * np.round(dy / box_yz[0])
            dz -= box_yz[1] * np.round(dz / box_yz[1])
            d = np.sqrt(dy**2 + dz**2)
            if d < r_max:
                distances.append(d)

    distances = np.array(distances)
    bins = np.arange(0, r_max + dr, dr)
    hist, edges = np.histogram(distances, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])

    area = box_yz[0] * box_yz[1]
    density_2d = N / area
    ideal_counts = 2 * np.pi * r * dr * density_2d * N / 2
    g2d = hist / ideal_counts
    return r, g2d

##############################################
# ④ 主循环：逐帧逐层
##############################################
all_layers = ["W", "X", "Y", "Z"]
g2d_accum = {layer: [] for layer in all_layers}

print("[INFO] Starting main loop over frames...")
for frame_idx, atoms in enumerate(trajectory):
    if frame_idx not in layer_ranges_per_frame:
        continue
    layer_ranges = layer_ranges_per_frame[frame_idx]

    x_coords = atoms.positions[:, 0]

    # 从cell读取盒子尺寸（每帧都有自己的盒子）
    box_y = atoms.cell.lengths()[1]
    box_z = atoms.cell.lengths()[2]
    box_yz = (box_y, box_z)

    for layer in all_layers:
        x_min, x_max = layer_ranges[layer]
        mask = (x_coords >= x_min) & (x_coords < x_max)
        selected = atoms.positions[mask]
        if len(selected) < 2:
            continue

        # 投影到 yz 平面
        positions2d = selected[:, 1:3]
        r, g2d = compute_g2d(positions2d, r_max, dr, box_yz)
        if g2d is not None:
            g2d_accum[layer].append(g2d)

print("[INFO] Finished accumulating RDFs over all frames.")

##############################################
# ⑤ 对所有帧平均
##############################################
average_g2d = {}
for layer in all_layers:
    data = np.array(g2d_accum[layer])
    if len(data) == 0:
        print(f"[WARN] No data for layer {layer}")
        continue
    avg = np.mean(data, axis=0)
    average_g2d[layer] = avg

##############################################
# ⑥ 绘图
##############################################
plt.figure(figsize=(8, 6))
for layer in all_layers:
    if layer in average_g2d:
        plt.plot(r, average_g2d[layer], label=f"Layer {layer}")

plt.xlabel("r (Å)", fontsize=14)
plt.ylabel("g₂D(r)", fontsize=14)
plt.title("2D RDF in 100 plane for WXYZ layers", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Plot image saved to: {output_image_file}")

# 也可以可视化显示（可选，取消注释即可）
# plt.show()

plt.close()

# ⑦ 保存数据
print("[INFO] Saving g2D(r) data to file...")
with open(output_data_file, 'w') as f:
    # 写标题
    header_line = "r"
    for layer in all_layers:
        header_line += f"\tg2D_{layer}"
    f.write(header_line + "\n")

    # 写数据
    for i in range(len(r)):
        line = f"{r[i]:.6f}"
        for layer in all_layers:
            if layer in average_g2d:
                line += f"\t{average_g2d[layer][i]:.6f}"
            else:
                line += f"\tNaN"
        f.write(line + "\n")

print(f"[SUCCESS] Data saved to: {output_data_file}")

print("[SUCCESS] All done!")

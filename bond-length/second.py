import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\100\cal_result\merge.xyz'
x_min, x_max = 88.4242,88.4242+3
frame_start, frame_end = 380, 400

# 键长过滤范围修改为 1.8-3.0 Å
bond_min, bond_max = 1.8, 3.0

# 手动盒子设置（与 S(q) 脚本保持一致）
MANUAL_BOX = None
MANUAL_LY_LZ = (53.6126, 53.6126)

# 输出
output_prefix = '111'
os.makedirs(output_prefix, exist_ok=True)
# -------------------------------------------------

# ---------- load trajectory ----------
u = mda.Universe(trajectory_file, format='XYZ')

# 设置盒子信息
if u.dimensions is None or np.allclose(u.dimensions[:3], 0):
    if MANUAL_BOX is not None:
        Lx, Ly, Lz = MANUAL_BOX
    elif MANUAL_LY_LZ is not None:
        Ly, Lz = MANUAL_LY_LZ
        u.trajectory[0]
        coords0 = u.atoms.positions.copy()
        Lx_est = coords0[:, 0].max() - coords0[:, 0].min()
        if Lx_est <= 0.0:
            Lx_est = max(Ly, Lz)
        Lx = Lx_est
    else:
        raise ValueError("请设置 MANUAL_BOX 或 MANUAL_LY_LZ")
    u.dimensions = [Lx, Ly, Lz, 90, 90, 90]
Lx, Ly, Lz = u.dimensions[:3]
print(f"使用盒子尺寸: Lx={Lx:.6f}, Ly={Ly:.6f}, Lz={Lz:.6f}")

# ---------- bond length analysis ----------
frames_to_use = range(frame_start, frame_end+1) if frame_end is not None else range(len(u.trajectory))
all_bond_lengths = []

for i, ts in enumerate(u.trajectory):
    if i not in frames_to_use:
        continue
    box = ts.dimensions
    inner_atoms = u.select_atoms(f'prop x >= {x_min} and prop x < {x_max}')
    if len(inner_atoms) == 0:
        continue
    dist_matrix = distance_array(inner_atoms.positions, u.atoms.positions, box=box)
    mask = dist_matrix > 0.1  # 去掉自己
    all_bond_lengths.extend(dist_matrix[mask])

# ---------- 后处理 ----------
all_bond_lengths = np.array(all_bond_lengths)
bond_lengths_filtered = all_bond_lengths[(all_bond_lengths >= bond_min) & (all_bond_lengths <= bond_max)]
print(f"总共统计到 {len(bond_lengths_filtered)} 个键长 (范围 {bond_min}-{bond_max} Å)")

# ---------- 归一化直方图 ----------
bins = 100
hist, bin_edges = np.histogram(bond_lengths_filtered, bins=bins, range=(bond_min, bond_max), density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
integral = np.trapz(hist, bin_centers)
print(f"归一化检查：积分 = {integral:.3f} (应接近 1.0)")

# ---------- 绘图 ----------
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
        alpha=0.8, color='darkcyan', edgecolor='black')
plt.xlabel('Bond Length (Å)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title(f'Normalized Bond Length Distribution\nAtoms in x ∈ [{x_min}, {x_max}) Å', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pngfile = f'{output_prefix}/{output_prefix}_x{x_min:.2f}_{x_max:.2f}.png'
plt.savefig(pngfile, dpi=300, bbox_inches='tight')
plt.show()
print("直方图保存为:", pngfile)

# ---------- 保存数据 ----------
datafile_raw = f'{output_prefix}/{output_prefix}_x{x_min:.2f}_{x_max:.2f}_raw.npy'
datafile_hist = f'{output_prefix}/{output_prefix}_x{x_min:.2f}_{x_max:.2f}_normalized_hist.txt'

np.save(datafile_raw, bond_lengths_filtered)
np.savetxt(datafile_hist,
           np.column_stack([bin_centers, hist]),
           header='Bin_center(Å)\tProbability_density')

print("数据保存完成！")
print("原始键长数据:", datafile_raw)
print("归一化直方图数据:", datafile_hist)


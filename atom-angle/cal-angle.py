import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from itertools import combinations
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\111\cal_result\merge.xyz'

# 切片范围（沿 X）
x_min = 85.3
x_max = 88.3

# 盒子信息
# 你之前手动设置的盒子尺寸
MANUAL_LY_LZ = (60.8935, 52.7353)   # (Ly, Lz)
MANUAL_BOX = None

# 帧选择
frame_start = 380
frame_end   = 400

# 成键距离截断
bond_cutoff = 2.85  # Å，根据体系调整

# 输出文件夹
output_dir = "111"
os.makedirs(output_dir, exist_ok=True)

# -------------------- 载入轨迹 --------------------
u = mda.Universe(trajectory_file, format="XYZ")

# 设置盒子尺寸
if u.dimensions is None:
    if MANUAL_BOX is not None:
        Lx, Ly, Lz = MANUAL_BOX
        u.dimensions = [Lx, Ly, Lz, 90, 90, 90]
    elif MANUAL_LY_LZ is not None:
        Ly, Lz = MANUAL_LY_LZ
        u.trajectory[0]
        coords0 = u.atoms.positions.copy()
        Lx_est = coords0[:, 0].max() - coords0[:, 0].min()
        if Lx_est <= 0.0:
            Lx_est = max(Ly, Lz)
        u.dimensions = [Lx_est, Ly, Lz, 90, 90, 90]
        u.trajectory.rewind()
    else:
        raise ValueError("请设置 MANUAL_BOX 或 MANUAL_LY_LZ")

Lx, Ly, Lz = u.dimensions[:3]
print(f"使用盒子尺寸: Lx={Lx:.6f}, Ly={Ly:.6f}, Lz={Lz:.6f}")

frames_to_use = set(range(frame_start, frame_end + 1))
all_angles = []

# -------------------- 遍历轨迹 --------------------
for i, ts in enumerate(u.trajectory):
    if i not in frames_to_use:
        continue

    # PBC wrap
    u.atoms.wrap(compound="atoms", inplace=True)
    coords = u.atoms.positions
    box = np.array([Lx, Ly, Lz])

    # Step 1: 选择切片内的中心原子（沿 X）
    mask = (coords[:, 0] >= x_min) & (coords[:, 0] < x_max)
    if mask.sum() == 0:
        continue
    center_indices = np.where(mask)[0]
    center_positions = coords[mask]

    for idx_center, pos_A in zip(center_indices, center_positions):
        # Step 2: 计算与所有原子的最小镜像向量
        dist_vec = coords - pos_A
        for k in range(3):
            dist_vec[:, k] -= box[k] * np.round(dist_vec[:, k] / box[k])
        dist = np.linalg.norm(dist_vec, axis=1)

        # 近邻索引
        neighbors = np.where((dist < bond_cutoff) & (dist > 0.1))[0]
        if len(neighbors) < 2:
            continue

        # Step 3: 生成所有角 B-A-C
        for j, k in combinations(neighbors, 2):
            AB = dist_vec[j]
            AC = dist_vec[k]
            cos_theta = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_theta))
            all_angles.append(theta)

all_angles = np.array(all_angles)

# -------------------- 绘图 --------------------
plt.figure(figsize=(9, 5))
plt.hist(all_angles, bins=90, range=(60, 180), alpha=0.8, color='orangered', edgecolor='black')
plt.xlabel('Bond Angle (°)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Bond Angle Distribution — x ∈ [{x_min}, {x_max}) Å (frames={frame_start}-{frame_end})', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

png_path = os.path.join(output_dir, f"bond_angle_xslice_{x_min:.1f}_{x_max:.1f}.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.show()
print("键角分布图保存为:", png_path)

np.save(os.path.join(output_dir, f"bond_angles_xslice_{x_min:.1f}_{x_max:.1f}.npy"), all_angles)
print("角度数据保存完成。")

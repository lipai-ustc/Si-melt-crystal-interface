import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\111\cal_result\merge.xyz'

# 切片范围（沿 X 方向，界面在 yz 平面）
x_min = 88.328
x_max = 88.328+3

# 帧选择
frame_start = 380
frame_end = 400   # 或 None 表示所有帧

# 键长过滤范围（避免长程噪声）
bond_min, bond_max = 0.8, 3  # Å

# 手动盒子设置（与 S(q) 脚本保持一致）
# 方式1：如果知道完整盒子大小 (Lx, Ly, Lz)
MANUAL_BOX = None
# 方式2：如果只知道 Ly, Lz，自动推测 Lx
MANUAL_LY_LZ = (60.8935, 52.7353)

# 输出
output_prefix = '111'
os.makedirs('111', exist_ok=True)
# -------------------------------------------------


# ---------- load trajectory ----------
u = mda.Universe(trajectory_file, format='XYZ')

# 设置盒子信息（借鉴 S(q) 脚本）
if u.dimensions is None or np.allclose(u.dimensions[:3], 0):
    if MANUAL_BOX is not None:
        Lx, Ly, Lz = MANUAL_BOX
        u.dimensions = [Lx, Ly, Lz, 90, 90, 90]
    elif MANUAL_LY_LZ is not None:
        Ly, Lz = MANUAL_LY_LZ
        u.trajectory[0]  # 先读取一帧
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

# ---------- bond length analysis ----------
all_bond_lengths = []

frames_to_use = range(frame_start, frame_end+1) if frame_end is not None else range(len(u.trajectory))

for i, ts in enumerate(u.trajectory):
    if i not in frames_to_use:
        continue

    # ✅ 必须传完整的 6 元素 box
    box = ts.dimensions

    # Step 1: 选择“切片内的原子”
    inner_atoms = u.select_atoms(f'prop x >= {x_min} and prop x < {x_max}')
    if len(inner_atoms) == 0:
        continue

    inner_positions = inner_atoms.positions
    all_positions = u.atoms.positions

    # Step 2: 计算切片内原子到全系统原子的最小镜像距离
    dist_matrix = distance_array(inner_positions, all_positions, box=box)

    # Step 3: 去掉“自己” (0 距离)，保留其他距离
    for j in range(len(inner_atoms)):
        dists = dist_matrix[j]
        valid_dists = dists[dists > 0.1]  # 过滤掉 self-distance
        all_bond_lengths.extend(valid_dists)

# ---------- 后处理 ----------
all_bond_lengths = np.array(all_bond_lengths)

# 过滤范围
bond_lengths_filtered = all_bond_lengths[(all_bond_lengths >= bond_min) & (all_bond_lengths <= bond_max)]

print(f"总共统计到 {len(bond_lengths_filtered)} 个键长 (范围 {bond_min}-{bond_max} Å)")

# ---------- 绘图 ----------
plt.figure(figsize=(8,5))
plt.hist(bond_lengths_filtered, bins=100, range=(bond_min, bond_max),
         alpha=0.8, color='darkcyan', edgecolor='black')
plt.xlabel('Bond Length (Å)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Bond Length Distribution\nAtoms in x ∈ [{x_min}, {x_max}) Å', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pngfile = f'111/{output_prefix}_x{x_min:.2f}_{x_max:.2f}.png'
plt.savefig(pngfile, dpi=300, bbox_inches='tight')
plt.show()
print("直方图保存为:", pngfile)

# 保存数据
np.save(f'111/{output_prefix}_x{x_min:.2f}_{x_max:.2f}.npy', bond_lengths_filtered)
np.savetxt(f'111/{output_prefix}_x{x_min:.2f}_{x_max:.2f}.txt',
           bond_lengths_filtered, header='Bond_length(Å)')
print("数据保存完成！")

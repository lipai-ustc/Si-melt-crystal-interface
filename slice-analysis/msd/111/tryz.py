import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\all_new_pot\111\msd\diffusion\diffusion.xyz'

# slice 基准位置
slice_center = 88.33
# 六个 slice 区间定义 (x_min, x_max)
slice_ranges = [
    (slice_center + 20, slice_center + 23),
    (slice_center, slice_center + 3),
    (slice_center - 3, slice_center),
    (slice_center - 6, slice_center - 3),
    (slice_center - 9, slice_center - 6),
    (slice_center - 40, slice_center - 37)
]

# 帧选择
frame_start = 380
frame_end = 400  # 或 None 表示所有帧

# 输出设置
output_prefix = '111_diffusion'
os.makedirs('111', exist_ok=True)
# -------------------------------------------------

# ====== Step 1: 读取 .xyz 文件（带 D_z 属性） ======
def load_xyz_with_Dz(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            try:
                num_atoms_line = f.readline()
                if not num_atoms_line:
                    break
                num_atoms = int(num_atoms_line)
                comment = f.readline()  # frame comment
                atoms = []
                for _ in range(num_atoms):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    # pos: x, y, z; D_x, D_y, D_z
                    x, y, z, D_x, D_y, D_z = map(float, parts[1:])
                    atoms.append((x, y, z, D_x, D_y, D_z))
                if atoms:
                    frames.append(np.array(atoms))
            except:
                break
    return frames

# ---------- 主程序 ----------
frames = load_xyz_with_Dz(trajectory_file)
print(f"总共读取到 {len(frames)} 帧，每帧 {frames[0].shape[0]} 个原子")

frames_to_use = range(frame_start, frame_end + 1) if frame_end is not None else range(len(frames))

for idx, (x_min, x_max) in enumerate(slice_ranges, 1):
    all_Dz = []
    for i in frames_to_use:
        frame = frames[i]
        x = frame[:, 0]
        Dz = frame[:, 5]  # D_z 列
        mask = (x >= x_min) & (x < x_max)
        if np.any(mask):
            all_Dz.extend(Dz[mask])
    all_Dz = np.array(all_Dz)

    # ---------- 计算均值 ----------
    if len(all_Dz) > 0:
        Dz_mean = np.mean(all_Dz)
        print(f"Slice {idx}: x ∈ [{x_min:.2f}, {x_max:.2f}) Å, "
              f"统计到 {len(all_Dz)} 个原子的 D_z, 均值 = {Dz_mean:.5f}")
    else:
        print(f"Slice {idx}: x ∈ [{x_min:.2f}, {x_max:.2f}) Å, 没有原子")

    # ---------- 绘图 ----------
    plt.figure(figsize=(8, 5))
    plt.hist(all_Dz, bins=100, alpha=0.8, color='steelblue', edgecolor='black')
    plt.xlabel('D_z', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'D_z Distribution\nAtoms in x ∈ [{x_min:.2f}, {x_max:.2f}) Å', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pngfile = f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.png'
    plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()
    print("直方图保存为:", pngfile)

    # 保存数据
    np.save(f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.npy', all_Dz)
    np.savetxt(f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.txt', all_Dz, header='D_z')
    print("数据保存完成！\n")

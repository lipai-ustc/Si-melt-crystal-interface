import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\111\cal_result\merge.xyz'

# slice 基准位置
slice_center = 88.33

# 六个 slice 区间定义 (x_min, x_max)
slice_ranges = [
    (slice_center + 20,  slice_center + 23),
    (slice_center, slice_center + 3),
    (slice_center - 3, slice_center),
    (slice_center - 6, slice_center - 3),
    (slice_center - 9, slice_center - 6),
    (slice_center - 40, slice_center - 37)
]

# 帧选择
frame_start = 380
frame_end = 400   # 或 None 表示所有帧

# 输出设置
output_prefix = '111_energy'   # 改成 energy
os.makedirs('111', exist_ok=True)
# -------------------------------------------------

# ====== Step 1: 读取 .xyz 文件 ======
def load_xyz_with_energy(filename):
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
                    # x, y, z, lq6, enthalpy, energy
                    x, y, z, lq6, enthalpy, energy = map(float, parts[1:])
                    atoms.append((x, y, z, lq6, enthalpy, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames

# ---------- 主程序 ----------
frames = load_xyz_with_energy(trajectory_file)
print(f"总共读取到 {len(frames)} 帧，每帧 {frames[0].shape[0]} 个原子")

frames_to_use = range(frame_start, frame_end + 1) if frame_end is not None else range(len(frames))

for idx, (x_min, x_max) in enumerate(slice_ranges, 1):
    all_energy = []

    for i in frames_to_use:
        frame = frames[i]
        x = frame[:, 0]
        energy = frame[:, 5]  # energy 列

        mask = (x >= x_min) & (x < x_max)
        if np.any(mask):
            all_energy.extend(energy[mask])

    all_energy = np.array(all_energy)
    print(f"Slice {idx}: x ∈ [{x_min:.2f}, {x_max:.2f}) Å, 统计到 {len(all_energy)} 个原子的 energy")

    # ---------- 绘图 ----------
    plt.figure(figsize=(8, 5))
    plt.hist(all_energy, bins=100,
             alpha=0.8, color='seagreen', edgecolor='black')
    plt.xlabel('Energy', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Energy Distribution\nAtoms in x ∈ [{x_min:.2f}, {x_max:.2f}) Å', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pngfile = f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.png'
    plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()
    print("直方图保存为:", pngfile)

    # 保存数据
    np.save(f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.npy', all_energy)
    np.savetxt(f'111/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.txt',
               all_energy, header='energy')
    print("数据保存完成！\n")

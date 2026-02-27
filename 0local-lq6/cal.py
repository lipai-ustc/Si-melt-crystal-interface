import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\110\cal_result\merge.xyz'

# slice 基准位置
slice_center = 89.84

# 六个 slice 区间定义 (x_min, x_max)
slice_ranges = [
    (slice_center + 20,  slice_center+ 23),
    (slice_center, slice_center + 3),
    (slice_center - 3, slice_center),
    (slice_center - 6, slice_center - 3),
    (slice_center - 9, slice_center - 6),
    (slice_center - 40,slice_center - 37)
]

# 帧选择
frame_start = 380
frame_end = 400   # 或 None 表示所有帧

output_prefix = '110_lq6'
os.makedirs('110', exist_ok=True)
# -------------------------------------------------

# ====== Step 1: 读取 .xyz 文件 ======
def load_xyz_with_lq6(filename):
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
                    x, y, z, lq6, entropy, energy = map(float, parts[1:])
                    atoms.append((x, y, z, lq6, entropy, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames

# ---------- 主程序 ----------
frames = load_xyz_with_lq6(trajectory_file)
print(f"总共读取到 {len(frames)} 帧，每帧 {frames[0].shape[0]} 个原子")

frames_to_use = range(frame_start, frame_end+1) if frame_end is not None else range(len(frames))

for idx, (x_min, x_max) in enumerate(slice_ranges, 1):
    all_lq6 = []

    for i in frames_to_use:
        frame = frames[i]
        x = frame[:, 0]
        lq6 = frame[:, 3]

        mask = (x >= x_min) & (x < x_max)
        if np.any(mask):
            all_lq6.extend(lq6[mask])

    all_lq6 = np.array(all_lq6)
    print(f"Slice {idx}: x ∈ [{x_min:.2f}, {x_max:.2f}) Å, 统计到 {len(all_lq6)} 个原子的 lq6")

    # ---------- 绘图 ----------
    plt.figure(figsize=(8,5))
    plt.hist(all_lq6, bins=100, range=(-0.2, 1),
             alpha=0.8, color='darkorange', edgecolor='black')
    plt.xlabel('lq6', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'lq6 Distribution\nAtoms in x ∈ [{x_min:.2f}, {x_max:.2f}) Å', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pngfile = f'110/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.png'
    plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()
    print("直方图保存为:", pngfile)

    # 保存数据
    np.save(f'110/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.npy', all_lq6)
    np.savetxt(f'110/{output_prefix}_slice{idx}_x{x_min:.2f}_{x_max:.2f}.txt',
               all_lq6, header='lq6')
    print("数据保存完成！\n")

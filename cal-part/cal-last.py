import numpy as np
import os

# ====== Step 1: 读取 .xyz 文件（含 lq6 和能量）======
def load_xyz_with_lq6_energy(filename):
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
                    atoms.append((x, y, z, lq6, energy))
                if atoms:
                    frames.append(np.array(atoms))
                else:
                    break
            except:
                break
    return frames


def main():
    xyz_file = r"E:\free_energy\0all_new_pot\110\cal_result\merge.xyz"
    output_dir = r"E:\free_energy\0all_new_pot\110\trypart_entropy_lq6color"
    os.makedirs(output_dir, exist_ok=True)

    y_min, y_max = 27, 40
    z_min, z_max = 15, 37
    frame_start, frame_end = 188, 189

    frames = load_xyz_with_lq6_energy(xyz_file)

    all_x, all_e, all_lq6 = [], [], []

    for i, frame in enumerate(frames):
        if i < frame_start or i > frame_end:
            continue

        x_atoms = frame[:, 0]
        y_atoms = frame[:, 1]
        z_atoms = frame[:, 2]
        lq6_values = frame[:, 3]
        energies = frame[:, 4]

        mask = (y_atoms > y_min) & (y_atoms < y_max) & (z_atoms > z_min) & (z_atoms < z_max)
        if np.sum(mask) == 0:
            continue

        all_x.append(x_atoms[mask])
        all_e.append(energies[mask])
        all_lq6.append(lq6_values[mask])

    if len(all_x) == 0:
        print("❌ 没有符合条件的原子，无法保存数据")
        return

    all_x = np.concatenate(all_x)
    all_e = np.concatenate(all_e)
    all_lq6 = np.concatenate(all_lq6)

    # 保存到 txt 文件
    save_path = os.path.join(output_dir, f"atom_data_frames_{frame_start}-{frame_end}.txt")
    data = np.column_stack([all_x, all_e, all_lq6])
    np.savetxt(save_path, data, header="x energy lq6", fmt="%.6f")
    print(f"[SUCCESS] 筛选后数据已保存: {save_path}")


if __name__ == "__main__":
    main()

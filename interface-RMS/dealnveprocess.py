from ase.io import read, write
from ase import Atoms
import numpy as np
import os
from ovito.io import import_file
from ovito.modifiers import IdentifyDiamondModifier

# === 用户设置 ===
input_file = r"D:\onedrive\Desktop\1"   # 原始多帧 XYZ 文件
start_frame = 0
end_frame   = 10    # 可以改成 len(轨迹) 或指定帧范围
x_min, x_max = 60, 100
bin_size = 8

# 输出设置
output_dir = "./out"
os.makedirs(output_dir, exist_ok=True)
output_xyz = os.path.join(output_dir, "interface_selected_atoms.xyz")

# === Step 0: 预处理文件，把所有原子先改成 Si ===
fixed_file = "fixed_input.xyz"
with open(input_file, "r") as fin, open(fixed_file, "w") as fout:
    lines = fin.readlines()
    i = 0
    while i < len(lines):
        natoms_line = lines[i]
        header_line = lines[i + 1]
        fout.write(natoms_line)
        fout.write(header_line)
        natoms = int(natoms_line.strip())
        for j in range(natoms):
            atom_line = lines[i + 2 + j].strip()
            parts = atom_line.split()
            # 改成 Si
            parts[0] = "Si"
            fout.write(" ".join(parts) + "\n")
        i += natoms + 2

# === Step 1: ASE 读取多帧轨迹 ===
ase_traj = list(read(fixed_file, format="extxyz", index=f"{start_frame}:{end_frame}"))
output_frames = []

for frame_index, ase_atoms in enumerate(ase_traj, start=start_frame):
    print(f"处理第 {frame_index} 帧 ...")

    ase_atoms.set_chemical_symbols(["Si"] * len(ase_atoms))

    # 临时写出供 OVITO 使用
    tmp_file = "tmp_for_ovito.xyz"
    write(tmp_file, ase_atoms, format="extxyz")

    pipeline = import_file(tmp_file)
    idm = IdentifyDiamondModifier()
    pipeline.modifiers.append(idm)
    data = pipeline.compute()

    structure_types = data.particles['Structure Type']
    positions = ase_atoms.positions

    # 只取 cubic diamond + 1st neighbor
    solid_mask = np.logical_or(
        structure_types == IdentifyDiamondModifier.Type.CUBIC_DIAMOND,
        structure_types == IdentifyDiamondModifier.Type.CUBIC_DIAMOND_FIRST_NEIGHBOR
    )
    solid_positions = positions[solid_mask]

    # 限定 x 范围
    mask_x = (solid_positions[:, 0] >= x_min) & (solid_positions[:, 0] <= x_max)
    solid_positions = solid_positions[mask_x]
    solid_indices = np.where(solid_mask)[0][mask_x]  # ASE 原子索引

    if len(solid_positions) == 0:
        output_frames.append(ase_atoms.copy())
        continue

    # Step 2: yz 分箱
    y, z = solid_positions[:, 1], solid_positions[:, 2]
    y_bins = np.arange(np.min(y), np.max(y) + bin_size, bin_size)
    z_bins = np.arange(np.min(z), np.max(z) + bin_size, bin_size)

    # 创建副本，用于修改选中原子为 S
    frame_atoms = ase_atoms.copy()

    for i in range(len(y_bins) - 1):
        for j in range(len(z_bins) - 1):
            mask_cell = (
                (solid_positions[:, 1] >= y_bins[i]) & (solid_positions[:, 1] < y_bins[i+1]) &
                (solid_positions[:, 2] >= z_bins[j]) & (solid_positions[:, 2] < z_bins[j+1])
            )
            cell_positions = solid_positions[mask_cell]
            cell_indices = solid_indices[mask_cell]

            if len(cell_positions) >= 3:
                x_sorted_idx = np.argsort(cell_positions[:, 0])
                selected_idx = cell_indices[x_sorted_idx[1:3]]  # 第二和第三最小 x 的原子
                # 修改选中的原子为 S
                for idx in selected_idx:
                    frame_atoms[idx].symbol = "S"

    output_frames.append(frame_atoms)

# === Step 3: 保存修改后的多帧 XYZ ===
write(output_xyz, output_frames, format="extxyz")
print(f"完成！结果保存在 {output_xyz}")

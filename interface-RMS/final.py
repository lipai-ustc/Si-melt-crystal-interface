import re
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ovito.io import import_file
from ovito.modifiers import IdentifyDiamondModifier

# === 用户设置 ===
input_file = r"E:\0try\110\npt\energy_extend.xyz"
start_frame = 0
end_frame   = 2000
x_min, x_max = 50, 90   # x 方向筛选范围
bin_size = 8              # yz 平面网格大小

# === 输出设置 ===
output_dir = r"E:\0try\110\npt\interface-rms"
os.makedirs(output_dir, exist_ok=True)
output_xyz = os.path.join(output_dir, "interface_atoms.xyz")
output_csv = os.path.join(output_dir, f"interface_variance-{bin_size}-{start_frame}-{end_frame}.csv")

# === Step 0: ASE 读取多帧轨迹并统一原子为 Si ===
ase_traj = list(read(input_file, format="extxyz", index=f"{start_frame}:{end_frame}"))
for frame in ase_traj:
    frame.set_chemical_symbols(["Si"] * len(frame))

output_frames = []
variances = []

def calc_bins(min_val, max_val, bin_size):
    """计算分箱边界，若最后一个箱长度不足一半则合并"""
    rng = max_val - min_val
    num_full_bins = int(rng // bin_size)
    remainder = rng - num_full_bins * bin_size

    bins = [min_val + i * bin_size for i in range(num_full_bins + 1)]
    if remainder >= bin_size / 2:
        bins.append(max_val)
    else:
        bins[-1] = max_val
    return np.array(bins)

# === Step 1: 处理每一帧 ===
for frame_index, ase_atoms in enumerate(ase_traj, start=start_frame):
    print(f"处理第 {frame_index} 帧 ...")

    tmp_file = "tmp_for_ovito.xyz"
    write(tmp_file, ase_atoms, format="extxyz")

    pipeline = import_file(tmp_file)
    idm = IdentifyDiamondModifier()
    pipeline.modifiers.append(idm)
    data = pipeline.compute()

    structure_types = data.particles['Structure Type']
    positions = ase_atoms.positions

    # 选择 cubic diamond + 1st neighbor 原子
    solid_mask = np.logical_or(
        structure_types == IdentifyDiamondModifier.Type.CUBIC_DIAMOND,
        structure_types == IdentifyDiamondModifier.Type.CUBIC_DIAMOND_FIRST_NEIGHBOR
    )
    solid_positions = positions[solid_mask]
    solid_indices = np.where(solid_mask)[0]

    # 限定 x 范围
    mask_x = (solid_positions[:, 0] >= x_min) & (solid_positions[:, 0] <= x_max)
    solid_positions = solid_positions[mask_x]
    solid_indices = solid_indices[mask_x]

    if len(solid_positions) == 0:
        variances.append(np.nan)
        output_frames.append(ase_atoms.copy())
        continue

    # Step 2: yz 平面划分网格（改进分箱逻辑）
    y, z = solid_positions[:, 1], solid_positions[:, 2]
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    y_bins = calc_bins(y_min, y_max, bin_size)
    z_bins = calc_bins(z_min, z_max, bin_size)

    frame_atoms = ase_atoms.copy()
    interface_x_values = []

    for i in range(len(y_bins) - 1):
        for j in range(len(z_bins) - 1):
            # y方向 mask
            y_mask = (solid_positions[:, 1] >= y_bins[i]) & (solid_positions[:, 1] < y_bins[i + 1])
            # z方向 mask
            z_mask = (solid_positions[:, 2] >= z_bins[j]) & (solid_positions[:, 2] < z_bins[j + 1])
            mask_cell = y_mask & z_mask

            cell_positions = solid_positions[mask_cell]
            cell_indices = solid_indices[mask_cell]

            if len(cell_positions) >= 3:
                x_sorted_idx = np.argsort(cell_positions[:, 0])
                selected_idx = cell_indices[x_sorted_idx[1:3]]  # 第二和第三最小 x 的原子
                interface_x_values.extend(cell_positions[x_sorted_idx[1:3], 0])
                for idx in selected_idx:
                    frame_atoms[idx].symbol = "S"

    # 计算界面方差
    if len(interface_x_values) > 0:
        variance = np.var(interface_x_values)
    else:
        variance = np.nan
        print(f"⚠️ 第 {frame_index} 帧没有足够原子计算界面方差")

    variances.append(variance)
    output_frames.append(frame_atoms)

# === Step 3: 保存修改后的多帧 XYZ ===
write(output_xyz, output_frames, format="extxyz")
print(f"界面原子标记 S 的多帧 XYZ 文件已保存：{output_xyz}")

# === Step 4: 保存界面方差 CSV ===
np.savetxt(output_csv, np.column_stack((np.arange(start_frame, end_frame), variances)),
           delimiter=",", header="frame,variance", comments="")
print(f"界面方差 CSV 已保存：{output_csv}")

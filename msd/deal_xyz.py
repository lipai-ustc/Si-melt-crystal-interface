from ase.io import read, write
import numpy as np

# === 参数配置 ===
input_file = r'E:\free_energy\0all_new_pot\100\msd\energy_extend.xyz'
output_file = r'E:\free_energy\0all_new_pot\100\msd\msd2_output.xyz'
frames = read(input_file, index=':')

n_atoms = len(frames[0])
n_frames = len(frames)

# 获取 origin，如果存在
origin = frames[0].info.get('Origin', [0.0, 0.0, 0.0])
origin = np.array([float(x) for x in origin])

output_frames = []

for i in range(n_frames):
    if i == 0:
        # 第一帧没有参考帧，位移设为零
        disp_vec = np.zeros((n_atoms, 3))
    else:
        ref_atoms = frames[i - 1]
        current_atoms = frames[i]

        ref_atoms.set_pbc([True, True, True])
        current_atoms.set_pbc([True, True, True])

        cell = current_atoms.get_cell()
        ref_pos = ref_atoms.get_positions() - origin
        cur_pos = current_atoms.get_positions() - origin
        disp_vec = cur_pos - ref_pos

        # 最小影像原则修正
        disp_vec -= np.round(disp_vec @ np.linalg.inv(cell)) @ cell

    # 分量平方
    disp_vec_sq = disp_vec ** 2

    atoms_with_disp = frames[i].copy()
    atoms_with_disp.new_array("disp_x2", disp_vec_sq[:, 0])
    atoms_with_disp.new_array("disp_y2", disp_vec_sq[:, 1])
    atoms_with_disp.new_array("disp_z2", disp_vec_sq[:, 2])
    output_frames.append(atoms_with_disp)

# 写入 extended XYZ 文件
write(output_file, output_frames, format='extxyz')

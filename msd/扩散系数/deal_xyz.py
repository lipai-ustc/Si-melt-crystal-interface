from ase.io import read, write
import numpy as np

# === 参数配置 ===
input_file = r'E:\free_energy\0all_new_pot\111\msd\energy_extend.xyz'
output_file = r'E:\free_energy\0all_new_pot\111\msd\diffusion\diffusion.xyz'
frames = read(input_file, index=':')

n_atoms = len(frames[0])
n_frames = len(frames)
dt_ps = 0.5  # 时间间隔 500 fs = 0.5 ps

# 原点坐标（可选）
origin = frames[0].info.get('Origin', [0.0, 0.0, 0.0])
origin = np.array([float(x) for x in origin])

output_frames = []

for i in range(n_frames):
    if i == 0:
        disp_vec = np.zeros((n_atoms, 3))
        D_x = np.zeros(n_atoms)
        D_y = np.zeros(n_atoms)
        D_z = np.zeros(n_atoms)
    else:
        ref_atoms = frames[i - 1]
        current_atoms = frames[i]

        ref_atoms.set_pbc([True, True, True])
        current_atoms.set_pbc([True, True, True])

        cell = current_atoms.get_cell()
        ref_pos = ref_atoms.get_positions() - origin
        cur_pos = current_atoms.get_positions() - origin
        disp_vec = cur_pos - ref_pos

        # 最小影像修正
        disp_vec -= np.round(disp_vec @ np.linalg.inv(cell)) @ cell

        # 分方向位移平方
        disp_x2 = disp_vec[:, 0] ** 2
        disp_y2 = disp_vec[:, 1] ** 2
        disp_z2 = disp_vec[:, 2] ** 2

        # 方向扩散系数 D = Δr² / (2Δt)
        D_x = disp_x2 / (2 * dt_ps)
        D_y = disp_y2 / (2 * dt_ps)
        D_z = disp_z2 / (2 * dt_ps)

    atoms_with_disp = frames[i].copy()
    atoms_with_disp.new_array("D_x", D_x)
    atoms_with_disp.new_array("D_y", D_y)
    atoms_with_disp.new_array("D_z", D_z)
    output_frames.append(atoms_with_disp)

# 输出
write(output_file, output_frames, format='extxyz')

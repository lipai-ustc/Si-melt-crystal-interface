from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import os

# === 0. 预处理 dump 文件 ===
traj_file = r'E:\free_energy\0all_new_pot\testmsd\equi_traj.lmp'  # <- 修改为你的轨迹文件

def ensure_type_field_in_dump(fname):
    try:
        # 尝试直接读取最后一帧
        _ = read(fname, index='-1', format='lammps-dump-text')
    except:
        # 如果失败，尝试修复缺失的 type 字段
        command = (
            f"grep type \"{fname}\" >nul 2>&1 || "
            f"(awk \"{{if(NF==4&&$3!=\\\"OF\\\") print $1,1,$2,$3,$4; "
            f"else if($3==\\\"id\\\") print \\\"ITEM: ATOMS id type x y z\\\"; else print}}\" \"{fname}\" > tmp && move /Y tmp \"{fname}\")"
        )
        os.system(command)

# 执行预处理
ensure_type_field_in_dump(traj_file)

# === 1. 读入轨迹 ===
frames = read(traj_file, index=':', format='lammps-dump-text')

n_atoms  = len(frames[0])
n_frames = len(frames)

# === 2. 取 Origin（若无则默认为 0）===
origin = np.array([float(x) for x in frames[0].info.get('Origin', [0.0, 0.0, 0.0])])

# === 3. 累积位移并做 PBC 解卷 ===
cum_disp = np.zeros((n_atoms, 3))
msd      = np.zeros(n_frames)

for i in range(1, n_frames):
    pos_prev = frames[i-1].get_positions() - origin
    pos_curr = frames[i  ].get_positions() - origin

    delta = pos_curr - pos_prev

    cell     = frames[i].get_cell()
    inv_cell = np.linalg.inv(cell)

    frac_disp = delta @ inv_cell
    delta    -= np.round(frac_disp) @ cell

    cum_disp += delta
    msd[i] = np.mean(np.sum(cum_disp**2, axis=1))

# === 4. 绘图 ===
timestep_ps = 1.0
time_axis   = np.arange(n_frames) * timestep_ps

plt.figure()
plt.plot(time_axis, msd, lw=1.5)
plt.xlabel('Time / ps')
plt.ylabel('MSD / Å$^2$')
plt.title('Mean Squared Displacement vs. Time')
plt.tight_layout()
plt.show()

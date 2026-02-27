import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from scipy import ndimage
import os

# -------------------- 用户参数 --------------------
trajectory_file = r'E:\free_energy\0all_new_pot\110\cal_result\merge.xyz'

# 切片范围（沿 X）
x_min = 89.845+6# 切片起始位置 (Å)
x_max = 89.845+9# 切片结束位置 (Å)

# 原始 histogram 网格
ny, nz = 256, 256        # 原始 histogram 网格
# 倒空间 FFT 输出网格（提高分辨率）
ny_fine, nz_fine = 1024, 1024

# 高斯平滑
smoothing_sigma = 0.0    # Å, 0 表示关闭

# 盒子信息（XYZ 文件一般没有）
#MANUAL_LY_LZ = (53.6126, 53.6126)  # (Ly, Lz)100
MANUAL_LY_LZ = (52.8256, 56.03)#110
#MANUAL_LY_LZ = (60.8935, 52.7353)#111
MANUAL_BOX = None                  # 或完整盒子 (Lx, Ly, Lz)

# 输出文件前缀
output_prefix = 'Sq_xslice'

# 帧选择
frame_list = None
frame_start = 380
frame_end = 400

# 归一化方式
normalization = 'meanN'  # 'meanN' 或 'instN'

# 径向平均
compute_radial = True
n_q_bins = 400
# -------------------------------------------------

# ---------- load trajectory ----------
u = mda.Universe(trajectory_file, format='XYZ')

# 设置盒子信息
if u.dimensions is None:
    if MANUAL_BOX is not None:
        Lx, Ly, Lz = MANUAL_BOX
        u.dimensions = [Lx, Ly, Lz, 90, 90, 90]
    elif MANUAL_LY_LZ is not None:
        Ly, Lz = MANUAL_LY_LZ
        u.trajectory[0]
        coords0 = u.atoms.positions.copy()
        Lx_est = coords0[:,0].max() - coords0[:,0].min()
        if Lx_est <= 0.0:
            Lx_est = max(Ly, Lz)
        u.dimensions = [Lx_est, Ly, Lz, 90, 90, 90]
        u.trajectory.rewind()
    else:
        raise ValueError("请设置 MANUAL_BOX 或 MANUAL_LY_LZ")

Lx, Ly, Lz = u.dimensions[0], u.dimensions[1], u.dimensions[2]
print(f"使用盒子尺寸: Lx={Lx:.6f}, Ly={Ly:.6f}, Lz={Lz:.6f}")

# 处理帧列表
if frame_list is not None:
    frames_to_use = set(frame_list)
else:
    frames_to_use = set(range(frame_start, frame_end+1)) if frame_start is not None else None

# 网格步长
dy = Ly / ny
dz = Lz / nz

# q 空间
q_y = np.fft.fftshift(np.fft.fftfreq(ny_fine, d=dy) * 2*np.pi)
q_z = np.fft.fftshift(np.fft.fftfreq(nz_fine, d=dz) * 2*np.pi)
Qy_mesh, Qz_mesh = np.meshgrid(q_y, q_z, indexing='ij')

# 累积
sum_F2 = np.zeros((ny_fine, nz_fine))
sum_instnorm = np.zeros((ny_fine, nz_fine))
sum_N = 0.0
frame_count = 0

# ---------- 遍历轨迹 ----------
for i, ts in enumerate(u.trajectory):
    if (frames_to_use is not None) and (i not in frames_to_use):
        continue
    if u.dimensions is not None:
        u.atoms.wrap(compound='atoms', inplace=True)
    coords = u.atoms.positions

    # X 切片选择（使用 x_min 和 x_max）
    mask = (coords[:,0] >= x_min) & (coords[:,0] < x_max)
    N_frame = mask.sum()
    if N_frame==0:
        continue
    yz = coords[mask,1:3]

    # histogram
    hist,_ ,_ = np.histogram2d(yz[:,0], yz[:,1], bins=(ny,nz), range=[[0,Ly],[0,Lz]])
    # 高斯平滑
    if smoothing_sigma>0:
        sigma_y = smoothing_sigma/dy
        sigma_z = smoothing_sigma/dz
        hist = ndimage.gaussian_filter(hist, sigma=(sigma_y, sigma_z), mode='wrap')

    # FFT + zero-padding
    F = np.fft.fft2(hist, s=(ny_fine, nz_fine))
    F_shift = np.fft.fftshift(F)
    F_abs2 = np.abs(F_shift)**2

    if normalization=='instN':
        sum_instnorm += F_abs2/N_frame
    else:
        sum_F2 += F_abs2
        sum_N += N_frame
    frame_count += 1

if frame_count==0:
    raise RuntimeError("没有找到任何原子在指定切片与帧范围内！")

# ---------- 计算平均 S(q) ----------
if normalization=='instN':
    S_q_avg = sum_instnorm/frame_count
else:
    S_q_avg = sum_F2/sum_N

# ---------- 径向平均 ----------
if compute_radial:
    q_mag = np.sqrt(Qy_mesh**2 + Qz_mesh**2)
    q_flat = q_mag.ravel()
    S_flat = S_q_avg.ravel()
    bins = np.linspace(0,q_flat.max(), n_q_bins+1)
    bin_idx = np.digitize(q_flat, bins)-1
    S_radial = np.zeros(n_q_bins)
    q_centers = 0.5*(bins[:-1]+bins[1:])
    for b in range(n_q_bins):
        maskb = (bin_idx==b)
        if maskb.sum()>0:
            S_radial[b] = S_flat[maskb].mean()
        else:
            S_radial[b] = np.nan

# ---------- 绘图与保存 (方法一: 裁剪 q 范围 ±12 Å^-1) ----------
os.makedirs('110', exist_ok=True)

# 设置倒空间显示范围
q_limit = 12.0  # Å^-1, ±范围

# 找到 q 对应的索引
iy_min = np.searchsorted(q_y, -q_limit)
iy_max = np.searchsorted(q_y, q_limit)
iz_min = np.searchsorted(q_z, -q_limit)
iz_max = np.searchsorted(q_z, q_limit)

# 裁剪数组
S_q_plot = S_q_avg[iy_min:iy_max, iz_min:iz_max]
qy_plot = q_y[iy_min:iy_max]
qz_plot = q_z[iz_min:iz_max]

# 绘制 2D 热图
plt.figure(figsize=(7,6))
plt.imshow(np.log10(S_q_plot + 1e-12).T,
           extent=[qy_plot[0], qy_plot[-1], qz_plot[0], qz_plot[-1]],
           origin='lower',
           aspect='equal',
           cmap='viridis')
plt.colorbar(label=r'$\log_{10}S(\mathbf{q})$')
plt.xlabel(r'$q_y\ (\AA^{-1})$')
plt.ylabel(r'$q_z\ (\AA^{-1})$')
plt.title(f'2D S(q)  — x ∈ [{x_min:.3f}, {x_max:.3f}] Å  (frames={frame_count})')
plt.tight_layout()
png2d = f'trycu/{output_prefix}_2D_x{x_min:.2f}_{x_max:.2f}_q{q_limit:.1f}.png'
plt.savefig(png2d, dpi=300, bbox_inches='tight')
plt.show()
print("2D 图保存为:", png2d)

# 保存裁剪后的数据
np.save(f'110/{output_prefix}_Sq2D_x{x_min:.2f}_{x_max:.2f}_q{q_limit:.1f}.npy', S_q_plot)


# 保存数据
np.save(f'110/{output_prefix}_Sq2D_x{x_min:.2f}_{x_max:.2f}.npy', S_q_avg)

# ---------- 径向平均图（裁剪 q 范围 2-8 Å^-1） ----------
if compute_radial:
    # 设置显示范围
    q_min_radial = 2.0  # Å^-1
    q_max_radial = 8.0  # Å^-1

    # 选取 q_centers 在范围内的索引
    radial_mask = (q_centers >= q_min_radial) & (q_centers <= q_max_radial)
    q_centers_plot = q_centers[radial_mask]
    S_radial_plot = S_radial[radial_mask]

    # 绘图
    plt.figure(figsize=(6,4))
    plt.plot(q_centers_plot, S_radial_plot, '-o', markersize=3)
    plt.xlabel(r'$q\ (\AA^{-1})$')
    plt.ylabel(r'$S(q)$')
    plt.title(f'Radial S(q) — x ∈ [{x_min:.3f}, {x_max:.3f}] Å (frames={frame_count})')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    pngrad = f'110/{output_prefix}_radial_x{x_min:.2f}_{x_max:.2f}_q{q_min_radial:.1f}-{q_max_radial:.1f}.png'
    plt.savefig(pngrad, dpi=300, bbox_inches='tight')
    plt.show()

    # 保存数据
    np.savetxt(f'110/{output_prefix}_radial_x{x_min:.2f}_{x_max:.2f}_q{q_min_radial:.1f}-{q_max_radial:.1f}.txt',
               np.vstack((q_centers_plot, S_radial_plot)).T,
               header='q S(q)')
    print("径向平均图保存为:", pngrad)

    """"# ---------- 绘图与保存 (完整 q 范围) ----------
    os.makedirs('111', exist_ok=True)

    # 不裁剪，直接使用整个 FFT 网格
    S_q_plot = S_q_avg
    qy_plot = q_y
    qz_plot = q_z

    # 绘制 2D 热图
    plt.figure(figsize=(7,6))
    plt.imshow(np.log10(S_q_plot + 1e-12).T,
               extent=[qy_plot[0], qy_plot[-1], qz_plot[0], qz_plot[-1]],
               origin='lower',
               aspect='equal',
               cmap='viridis')
    plt.colorbar(label=r'$\log_{10}S(\mathbf{q})$')
    plt.xlabel(r'$q_y\ (\AA^{-1})$')
    plt.ylabel(r'$q_z\ (\AA^{-1})$')
    plt.title(f'2D S(q)  — x ∈ [{x_min:.3f}, {x_max:.3f}] Å  (frames={frame_count})')
    plt.tight_layout()
    png2d = f'111/{output_prefix}_2D_x{x_min:.2f}_{x_max:.2f}_full.png'
    plt.savefig(png2d, dpi=300, bbox_inches='tight')
    plt.show()
    print("2D 图保存为:", png2d)
    """


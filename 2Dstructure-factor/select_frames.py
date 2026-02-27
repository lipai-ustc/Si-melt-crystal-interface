import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
from scipy import ndimage

# =================== 参数设置 ===================
trajectory_file = r'E:\free_energy\0all_new_pot\100\cal_result\merge.xyz'   # XYZ 文件路径
x_slice = 10                      # 想要分析的 X 高度（Å）
dx_slice = 1.4                           # 切片厚度（Å）
ny, nz = 256, 256                        # 2D 网格分辨率（YZ 平面），建议用 2 的幂
sigma_smooth = 0.8                       # 高斯平滑核大小（Å）

# 如果 XYZ 文件不包含盒子信息，请手动设置 Ly, Lz
MANUAL_LY_LZ = (53.6126, 53.6126)        # ⚠️ 修改为你的盒子大小 (Ly, Lz)

# 输出图像文件名
output_file = f'structure_factor_x{x_slice:.1f}.png'

# ===== 帧选择设置 =====
frame_start = 390      # 起始帧编号（从 0 开始）
frame_end   = 400     # 结束帧编号（包含）
# 如果想处理所有帧，设置 frame_start=None, frame_end=None
# ================================================

# 加载轨迹
u = mda.Universe(trajectory_file, format='XYZ')

# 如果 XYZ 没有盒子信息，就用手动设置
if MANUAL_LY_LZ is not None:
    Ly_manual, Lz_manual = MANUAL_LY_LZ
else:
    if u.dimensions is None:
        raise ValueError("轨迹中没有盒子信息，请设置 MANUAL_LY_LZ")
    Ly_manual, Lz_manual = None, None

# 初始化累积的 |F|²
S_q_sum = np.zeros((ny, nz))
frame_count = 0

# 遍历轨迹
for i, ts in enumerate(u.trajectory):
    # 帧范围选择
    if (frame_start is not None and i < frame_start) or \
       (frame_end   is not None and i > frame_end):
        continue

    # 如果有盒子信息，可以 wrap
    if u.dimensions is not None:
        u.atoms.wrap(compound='atoms', inplace=True)

    # 获取盒子尺寸
    if u.dimensions is not None:
        Ly, Lz = u.dimensions[1], u.dimensions[2]
    else:
        Ly, Lz = Ly_manual, Lz_manual

    # 网格步长
    dy = Ly / ny
    dz = Lz / nz

    # q 空间坐标
    q_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi  # Å⁻¹
    q_z = np.fft.fftshift(np.fft.fftfreq(nz, d=dz)) * 2 * np.pi
    Qy, Qz = np.meshgrid(q_y, q_z)

    # ====== 选择 X 切片内的原子 ======
    x_low = x_slice - dx_slice / 2
    x_high = x_slice + dx_slice / 2
    sel = u.select_atoms(f'prop x >= {x_low} and prop x < {x_high}')
    if len(sel) == 0:
        continue  # 没有原子则跳过

    # ====== 构建 2D 密度场 ρ(y,z) ======
    yz = sel.positions[:, 1:3]  # 取 Y,Z 坐标

    hist, yedges, zedges = np.histogram2d(
        yz[:, 0], yz[:, 1],
        bins=(ny, nz),
        range=[[0, Ly], [0, Lz]]
    )

    # 高斯平滑
    rho_yz = ndimage.gaussian_filter(hist, sigma=sigma_smooth / dy)

    # ====== 2D FFT → 结构因子 S(q) ======
    F_rho = np.fft.fft2(rho_yz)
    F_rho = np.fft.fftshift(F_rho)          # 零频移到中心
    S_q = np.abs(F_rho)**2                  # S(q) = |ρ(q)|²

    # 累加
    S_q_sum += S_q
    frame_count += 1

# ====== 时间平均 ======
if frame_count == 0:
    raise ValueError(f"在 x ∈ [{x_low}, {x_high}) 范围内没有找到任何原子。")

S_q_avg = S_q_sum / frame_count

# ====== 绘图 ======
plt.figure(figsize=(9, 8))

S_q_plot = np.log10(S_q_avg + 1)  # 对数增强对比度

im = plt.imshow(
    S_q_plot,
    extent=[q_y[0], q_y[-1], q_z[0], q_z[-1]],
    cmap='viridis',
    origin='lower',
    aspect='equal'
)

# 颜色条
cbar = plt.colorbar(im, pad=0.02)
cbar.set_label(r'$\log_{10}(S(\mathbf{q}) + 1)$', rotation=270, labelpad=25, fontsize=14)

# 坐标轴
plt.xlabel(r'$q_y$ (Å⁻¹)', fontsize=14)
plt.ylabel(r'$q_z$ (Å⁻¹)', fontsize=14)
plt.title(f'2D Structure Factor (Average over {frame_count} frames)\n'
          f'x = {x_slice:.1f} ± {dx_slice/2:.1f} Å', fontsize=14)

plt.tight_layout()
plt.show()

# ====== 保存结果 ======
if output_file:
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图像已保存为: {output_file}")

np.save(f'Sq_avg_x{x_slice:.1f}.npy', S_q_avg)
print(f"结构因子数据已保存为: Sq_avg_x{x_slice:.1f}.npy")

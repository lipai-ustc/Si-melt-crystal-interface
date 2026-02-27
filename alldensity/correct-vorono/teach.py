# density_comparison_with_voronoi_output.py
import numpy as np
from ase.io import read, write
import freud
from scipy import stats
import matplotlib.pyplot as plt

# ========== 参数设置 ==========
filename = r'E:\free_energy\0all_new_pot\100\cal_result\atoms.xyz'                    # 输入文件
output_xyz = r'E:\free_energy\0all_new_pot\100\density\correct\atoms_with_voronoi.xyz'     # 输出带体积信息的 XYZ 文件
n_frames = 10                             # 分析前 n 帧
n_bins = 500                            # x 方向 bin 数量
output_prefix = 'density_comparison'      # 图片和数据输出前缀

# ========== 数据收集 ==========
all_x_positions = []
all_geometric_densities = []
all_voronoi_volumes = []

print(f"开始分析 {filename} 的前 {n_frames} 帧，并保存 Voronoi 体积...")

# 存储处理后的原子对象（用于写入新 xyz）
frames_with_volume = []

for frame_idx in range(n_frames):
    try:
        atoms = read(filename, index=frame_idx)
        atoms.wrap()  # 确保原子在盒子内
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        x_positions = positions[:, 0]

        # ======== 1. 几何密度（切片）========
        yz_area = np.linalg.norm(np.cross(cell[1], cell[2]))
        bin_width = cell[0, 0] / n_bins
        bin_volume = bin_width * yz_area

        hist_counts, bin_edges_geom = np.histogram(x_positions, bins=n_bins, range=(0, cell[0, 0]))
        geometric_density_per_bin = hist_counts / bin_volume

        bin_indices = np.digitize(x_positions, bin_edges_geom) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        geometric_densities = geometric_density_per_bin[bin_indices]

        # ======== 2. Voronoi 体积计算 ========
        cell_matrix = np.asarray(cell)
        box = freud.box.Box.from_matrix(cell_matrix)
        voronoi = freud.locality.Voronoi()
        voronoi.compute((box, positions))
        volumes = np.array(voronoi.volumes)  # (N,) 每个原子的 Voronoi 体积

        # ✅ 将 Voronoi 体积写入原子 arrays
        atoms.arrays['voronoi_volume'] = volumes  # 可在 VMD/Ovito 中显示为 "Atomic Volume"

        # 保存这一帧（带体积信息）
        frames_with_volume.append(atoms.copy())

        # 收集数据用于统计
        all_x_positions.extend(x_positions)
        all_geometric_densities.extend(geometric_densities)
        all_voronoi_volumes.extend(volumes)

        print(f"第 {frame_idx} 帧处理完成: {len(volumes)} 个原子")

    except Exception as e:
        print(f"第 {frame_idx} 帧处理失败: {e}")

# ========== 保存带 Voronoi 体积的新 XYZ 文件 ==========
try:
    write(output_xyz, frames_with_volume)
    print(f"✅ 已保存带 Voronoi 体积的结构到: {output_xyz}")
    print("   💡 提示：可用 VMD/Ovito 打开，将 'voronoi_volume' 设为 'Color by' 查看分布")
except Exception as e:
    print(f"❌ 保存新 XYZ 文件失败: {e}")

# ========== 统计平均密度 ==========
all_x_positions = np.array(all_x_positions)
all_geometric_densities = np.array(all_geometric_densities)
all_voronoi_volumes = np.array(all_voronoi_volumes)

bin_edges = np.linspace(0, cell[0, 0], n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 几何密度平均
mean_geometric_density, _, _ = stats.binned_statistic(
    all_x_positions, all_geometric_densities, statistic='mean', bins=bin_edges
)

# Voronoi: 先平均体积，再取密度
mean_volumes, _, _ = stats.binned_statistic(
    all_x_positions, all_voronoi_volumes, statistic='mean', bins=bin_edges
)
mean_local_density = np.where(mean_volumes > 0, 1.0 / mean_volumes, np.nan)

# ========== 绘图 ==========
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_geometric_density,
         label='Geometric Density\n(from binning)', color='blue', linewidth=2)
plt.plot(bin_centers, mean_local_density,
         label='Local Density\n(from ⟨Voronoi Volume⟩)', color='red', linestyle='--', linewidth=2)
plt.xlabel('X Coordinate (Å)', fontsize=12)
plt.ylabel('Density (Å⁻³)', fontsize=12)
plt.title('Density Profile Comparison\nGeometric vs Voronoi (Smoothed)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, cell[0, 0])
plt.tight_layout()

# 保存图片
plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 密度图已保存为: {output_prefix}.png")

# ========== 保存数据 ==========
np.savetxt(
    f'{output_prefix}_data.txt',
    np.column_stack([bin_centers, mean_geometric_density, mean_local_density]),
    header='x_center(A)  geometric_density(atoms/A3)  local_density(1/<V>)',
    fmt='%.8f',
    delimiter='  '
)
print(f"💾 数据已保存为: {output_prefix}_data.txt")

import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 用户参数 =====
density_file = r"E:\free_energy\0all_new_pot\100\density\correct\binned_density.txt"
output_image = r"E:\free_energy\0all_new_pot\100\density\correct\secondbin.png"

plot_x_min = -20
plot_x_max = 20

# ===== 读取密度数据 =====
data = np.loadtxt(density_file, skiprows=1)
centers = data[:, 0]       # bin 中心
density = data[:, 1]       # 对应密度

# ===== 绘图 =====
plt.figure(figsize=(8,5))

# 选定范围
mask = (centers >= plot_x_min) & (centers <= plot_x_max)
centers_plot = centers[mask]
density_plot = density[mask]

plt.plot(centers_plot, density_plot, color='C0', alpha=0.9, linewidth=2)

plt.xlabel("z (Å)")
plt.ylabel("Number density (Å⁻³)")
plt.xlim(plot_x_min, plot_x_max)
plt.ylim(bottom=0)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()

print(f"Plot saved to: {output_image}")

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -------------------- 用户参数 --------------------
data_folder = r"E:\free_energy\MLSCAN\python_script\analysis\atom-angle\111"
bins = 100               # 直方图分箱数
angle_min, angle_max = 60, 180
sigma = 2                # 高斯平滑强度
output_folder = os.path.join(data_folder, "combined_plot")
output_figure = os.path.join(output_folder, "combined_angles_smoothed.png")

# 可以自定义每条线的名字，对应 npy 文件排序后的顺序
custom_labels = [
    "solid",
    "(0,+3)",
    "(-3,0)",
    "(-6,-3)",
    "(-9,-6)",
    "liquid",
  # ...根据实际文件数添加
]
# -------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 找到所有角度的 npy 文件
npy_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.endswith(".npy") and f.startswith("bond_angles_xslice")]

if not npy_files:
    raise FileNotFoundError("没有找到 bond_angles_xslice_*.npy 文件，请检查路径！")

plt.figure(figsize=(8, 5))
valid_files = 0

for idx, f in enumerate(sorted(npy_files)):
    try:
        angles = np.load(f)
        if angles.size == 0:
            print(f"跳过空文件: {f}")
            continue

        # ---- 归一化直方图 ----
        hist, bin_edges = np.histogram(angles, bins=bins,
                                       range=(angle_min, angle_max), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        integral = np.trapz(hist, bin_centers)
        print(f"{os.path.basename(f)}: 归一化检查 -> 积分 = {integral:.3f}")

        # ---- 高斯平滑 ----
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)

        # ---- 将概率乘以100 ----
        hist_smooth *= 100

        # ---- 自定义标签 ----
        if idx < len(custom_labels):
            label = custom_labels[idx]
        else:
            label = f"Slice {idx+1}"  # 自动生成标签

        plt.plot(bin_centers, hist_smooth, label=label, lw=2)
        valid_files += 1

    except Exception as e:
        print(f"读取文件 {f} 出错: {e}")
        continue

if valid_files == 0:
    raise RuntimeError("没有任何有效角度数据可以绘制！")

# ---- 坐标轴设置 ----
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))   # 主刻度间隔
ax.yaxis.set_minor_locator(AutoMinorLocator(3))
plt.xlabel("Angle (°)", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(angle_min, angle_max)
plt.ylim(0, 4.6)  # y轴范围，原0.045乘以100
plt.legend(fontsize=20, loc="best", frameon=False, shadow=False)
plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"多条曲线已绘制并保存到: {output_figure}")

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -------------------- 用户参数 --------------------
# D_x 的 npy 文件所在文件夹（即你上一步保存 111_Dx_slice*.npy 的地方）
data_folder = r"E:\free_energy\MLSCAN\python_script\analysis\0slice-analysis\msd\111"

bins = 100  # 直方图分箱数

# ⚠️ 根据实际 D_x 范围调整（这里假设 D_x 范围为 0 ~ 1，可根据你保存的数据改）
Dx_min, Dx_max = 0.0, 2

sigma = 2  # 高斯平滑强度
output_folder = os.path.join(data_folder, "combined_plot")
output_figure = os.path.join(output_folder, "combined_Dx.png")

# 自定义每条曲线的名字（对应 6 个 slice）
custom_labels = [
    "liquid (x+20~+23)",
    "slice0 (x+0~+3)",
    "slice-3_0",
    "slice-6_-3",
    "slice-9_-6",
    "solid (x-40~-37)"
]
# -------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 找到所有 *_Dx_slice*.npy 文件
npy_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.endswith(".npy") and "111" in f]

if not npy_files:
    raise FileNotFoundError("没有找到 D_x npy 文件，请检查路径！")

plt.figure(figsize=(8, 5))
valid_files = 0

# 按文件名排序，确保顺序与 custom_labels 一致
for idx, f in enumerate(sorted(npy_files)):
    try:
        Dx_data = np.load(f)
        if Dx_data.size == 0:
            print(f"跳过空文件: {f}")
            continue

        # ---- 归一化直方图 ----
        hist, bin_edges = np.histogram(Dx_data, bins=bins,
                                       range=(Dx_min, Dx_max), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        integral = np.trapz(hist, bin_centers)
        print(f"{os.path.basename(f)}: 归一化检查 -> 积分 = {integral:.3f}")

        # ---- 高斯平滑 ----
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)

        # ---- 标签 ----
        label = custom_labels[idx] if idx < len(custom_labels) else f"Slice {idx+1}"
        plt.plot(bin_centers, hist_smooth, label=label, lw=2)
        valid_files += 1

    except Exception as e:
        print(f"读取文件 {f} 出错: {e}")
        continue

if valid_files == 0:
    raise RuntimeError("没有任何有效 D_x 数据可以绘制！")

# ---- 坐标轴设置 ----
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.xlabel("D_x", fontsize=17)
plt.ylabel("Probability", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(Dx_min, Dx_max)
plt.ylim(0, 11)  # 可根据实际数据调整
plt.legend(fontsize=13, loc="best", frameon=False)
plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"多条归一化 D_x 曲线已绘制并保存到: {output_figure}")

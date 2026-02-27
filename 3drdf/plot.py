import numpy as np
import matplotlib.pyplot as plt
import os

# 输入文件路径
input_data_file = r"E:\free_energy\all_new_pot\110\3drdf\ABCDE_g3d_result.txt"

# 输出设置
output_dir = r"E:\free_energy\all_new_pot\110\3drdf\figs"
output_filename = "g3d_layers_corrected.png"
dpi = 400  # 可根据需要修改 DPI

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)

# 读取数据
data = np.loadtxt(input_data_file, skiprows=1)
r = data[:, 0]
g3d_layers = data[:, 1:]

# 图层设置
layer_names = ["A", "B", "C", "D", "liquid"]  # ABCD + 液体层
alphas = np.linspace(1, 0.4, len(layer_names))  # A 最亮，liquid 最暗

offset_step = 1.0
fs = 23  # 字体大小
plot_x_min, plot_x_max = 0, 10

# 创建绘图对象
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()

# 存储每条线和标签
lines = []
labels = []

# 绘图：A 偏移最大，liquid 最小
for i in range(len(layer_names)):
    offset = (len(layer_names) - 1 - i) * offset_step
    line, = ax.plot(r, g3d_layers[:, i] + offset,
                    color="C1", alpha=alphas[i], label=f"{layer_names[i]}")
    lines.append(line)
    labels.append(f"{layer_names[i]}")

# 坐标轴与标签设置
ax.set_xlabel("r (Å)", fontsize=fs)
ax.set_ylabel("g(r)", fontsize=fs)
ax.set_xlim(plot_x_min, plot_x_max)
ax.set_ylim(bottom=0)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# 图例顺序与视觉一致
ax.legend(lines, labels, fontsize=fs - 10)

plt.tight_layout()

# 保存图像
plt.savefig(output_path, dpi=dpi)
plt.close()

print(f"图像已保存到: {output_path}")

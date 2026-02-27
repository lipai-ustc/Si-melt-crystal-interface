import numpy as np
import matplotlib.pyplot as plt
import os
import re

# -------------------- 用户参数 --------------------
input_folder = r"111"  # 存放原始 txt 文件的文件夹
output_folder = r"sq111"  # 新图保存位置
output_prefix = "Sq_xslice"  # 和原始计算代码一致
custom_labels = ["solid", "+3", "0","-3","-6","liquid"]  # 这里可以自己改 legend 名称
marker_size = 2  # 缩小原点大小
# --------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 匹配文件名的正则表达式
pattern = re.compile(rf"{output_prefix}_radial_x([0-9.]+)_([0-9.]+)_q([0-9.]+)-([0-9.]+)\.txt")

# 新建一张图
plt.figure(figsize=(6, 4))

color_cycle = plt.cm.tab10.colors  # 取 10 种常用颜色
color_index = 0
label_index = 0

# 降序读取文件
for file in sorted(os.listdir(input_folder), reverse=True):
    match = pattern.match(file)
    if match:
        file_path = os.path.join(input_folder, file)

        # 读取数据
        data = np.loadtxt(file_path, comments='#')
        q = data[:, 0]
        S = data[:, 1]

        # 循环颜色
        color = color_cycle[color_index % len(color_cycle)]
        color_index += 1

        # 自定义图例（若超出 custom_labels 长度，使用文件名）
        if label_index < len(custom_labels):
            label = custom_labels[label_index]
        else:
            label = file.replace('.txt', '')
        label_index += 1

        # 绘制在同一张图上，调小 markersize
        plt.plot(q, S, '-', markersize=marker_size, label=label, color=color)

# 设置坐标轴和格式
plt.xlabel('q (Å⁻¹)', fontsize=16)
plt.ylabel('S(q)', fontsize=16)
plt.xlim(2.5, 8)
plt.ylim(0, 10)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=15, loc="best", frameon=False, shadow=False)
plt.tight_layout()

# 保存一张合并后的图
save_path = os.path.join(output_folder, f"{output_prefix}_combined.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"所有曲线已合并绘制并保存: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
import os
import re

# -------------------- 用户参数 --------------------
input_folder = r"100"  # 存放原始 txt 文件的文件夹
output_folder = r"sq100"  # 新图保存位置
output_prefix = "Sq_xslice"  # 和原始计算代码一致
# --------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 匹配文件名的正则表达式
pattern = re.compile(rf"{output_prefix}_radial_x([0-9.]+)_([0-9.]+)_q([0-9.]+)-([0-9.]+)\.txt")

for file in os.listdir(input_folder):
    match = pattern.match(file)
    if match:
        x_min, x_max, q_min, q_max = match.groups()
        file_path = os.path.join(input_folder, file)

        # 读取数据
        data = np.loadtxt(file_path, comments='#')
        q = data[:, 0]
        S = data[:, 1]

        # 重新绘图（无标题、调大字号）
        plt.figure(figsize=(6, 4))
        plt.plot(q, S, '-', markersize=3)
        plt.xlabel('q (Å⁻¹)', fontsize=16)
        plt.ylabel('S(q)', fontsize=16)
        plt.xlim(2.5, 8)  # <-- 在这里设置 y 轴范围
        plt.ylim(0, 9)  # <-- 在这里设置 y 轴范围
        plt.tick_params(axis='both', which='major', labelsize=14)
        #plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # 保存新图（文件名后加 _replot）
        save_name = file.replace('.txt', '_replot.png')
        save_path = os.path.join(output_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"重新绘制并保存: {save_path}")

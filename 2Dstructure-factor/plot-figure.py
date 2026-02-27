import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 原始文件夹和新输出文件夹
input_folder = "111"
output_folder = "111replot"
os.makedirs(output_folder, exist_ok=True)

# 搜索所有符合命名规则的 .npy 文件
npy_files = glob.glob(os.path.join(input_folder, "*_Sq2D_x*.npy"))

print(f"找到 {len(npy_files)} 个文件，开始重新绘图...")

for f in npy_files:
    data = np.load(f)

    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(data + 1e-12).T,
               origin='lower',
               aspect='equal',
               cmap='viridis')
 #   plt.colorbar(label=r'$\log_{10}S(\mathbf{q})$')

    # 去掉坐标轴刻度和标签
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")  # 不显示标题

    # 保存到新文件夹
    basename = os.path.basename(f).replace(".npy", ".png")
    save_path = os.path.join(output_folder, basename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {save_path}")

print("全部绘图完成！")

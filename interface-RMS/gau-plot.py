import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === 读取 CSV 文件 ===
csv_file = r"E:\0try\110\npt\interface-rms2\110.csv"
df = pd.read_csv(csv_file)

# 提取帧号和方差
frames = df['frame'].values
variances = df['variance'].values

# === 高斯平滑 ===
sigma = 4  # 平滑强度，可以调节（越大越平滑）
smoothed_variances = gaussian_filter1d(variances, sigma=sigma)

# === 保存平滑结果 ===
output_csv = csv_file.replace(".csv", f"-smoothed-sigma{sigma}.csv")
pd.DataFrame({'frame': frames, 'variance_smoothed': smoothed_variances}).to_csv(output_csv, index=False)
print(f"平滑后的数据已保存：{output_csv}")

# === 可选：绘图对比 ===
plt.figure(figsize=(8,4))
plt.plot(frames, variances, label='Original', alpha=0.5)
plt.plot(frames, smoothed_variances, label=f'Smoothed (σ={sigma})', linewidth=2)
plt.xlabel('Frame')
plt.ylabel('Variance')
plt.legend()
plt.title('Interface Variance with Gaussian Smoothing')
plt.tight_layout()
plt.show()

import os
import numpy as np
import pandas as pd

# ===== 配置部分 =====
folder_path = r"E:\0growth_velocity\0\111"  # 文件夹路径
N_threshold = 160  # 阈值，可自行修改
# ==================

# 获取文件夹下所有txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

all_data = []
velocities = []

for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过以 # 开头的行
    data_lines = [line.strip() for line in lines if not line.startswith('#')]

    # 读取数据
    data = np.array([list(map(float, line.split())) for line in data_lines])
    times = data[:, 0]  # 第一列是时间
    trajectories = data[:, 1:]  # 后面每列是轨迹

    all_data.append(data)

    # 计算每条轨迹的结晶速度
    file_velocities = []
    for traj in trajectories.T:
        t1_indices = np.where(traj >= N_threshold)[0]
        if len(t1_indices) == 0:
            t1 = np.nan
        else:
            t1 = times[t1_indices[0]]

        max_index = np.argmax(traj)
        t2 = times[max_index]
        Nmax = traj[max_index]

        if np.isnan(t1) or t2 == t1:
            velocity = np.nan
        else:
            velocity = (Nmax - N_threshold) / (t2 - t1)

        file_velocities.append(velocity)
    velocities.append(file_velocities)

# 合并所有文件的数据
final_times = all_data[0][:, 0]
final_trajs = np.hstack([d[:, 1:] for d in all_data])
final_velocities = np.concatenate(velocities)

# 计算平均结晶速度（忽略 nan）
average_velocity = np.nanmean(final_velocities)

# 创建 pandas DataFrame
traj_cols = [f"traj{i + 1}" for i in range(final_trajs.shape[1])]
vel_cols = [f"v{i + 1}" for i in range(final_trajs.shape[1])]

df_traj = pd.DataFrame(final_trajs, columns=traj_cols)
df_time = pd.DataFrame(final_times, columns=["Time"])
df_vel = pd.DataFrame([final_velocities] * len(final_times), columns=vel_cols)  # 每行填速度
df = pd.concat([df_time, df_traj, df_vel], axis=1)

print(df.head())
print(f"\n阈值 N_threshold = {N_threshold}")
print(f"所有轨迹的平均结晶速度：{average_velocity}")

# 如果想保存为 Excel 或 CSV，可以使用：
#df.to_excel(os.path.join(folder_path, "all.xlsx"), index=False)
df.to_csv(os.path.join(folder_path, "all.csv"), index=False)

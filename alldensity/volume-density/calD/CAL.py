import pandas as pd
import os
import glob

# 📁 设置你要处理的文件夹路径
root_folder = r"E:\free_energy\0all_new_pot\calD\density"

# 🔍 遍历所有TXT文件
for file_path in glob.glob(os.path.join(root_folder, "**", "*.txt"), recursive=True):
    try:
        # 读取TXT，按空白分隔，跳过注释
        df = pd.read_csv(file_path, delim_whitespace=True, comment="#", header=None)

        # 手动设置列名
        df.columns = ["Relative_x", "Smoothed_AtomicDensity"]

        if "Smoothed_AtomicDensity" not in df.columns or "Relative_x" not in df.columns:
            print(f"⚠️ 跳过 {file_path} ：缺少所需列")
            continue

        # 提取 S1 和 S_last
        S1 = df.iloc[0]["Smoothed_AtomicDensity"]
        S_last = df.iloc[-1]["Smoothed_AtomicDensity"]

        # 计算 D1 和 D2
        D1 = 0.90 * S1 + 0.1 * S_last
        D2 = 0.1 * S1 + 0.90 * S_last

        # 找到最接近 D1 和 D2 的 Smoothed_AtomicDensity 对应的 Relative_x
        idx_D1 = (df["Smoothed_AtomicDensity"] - D1).abs().idxmin()
        idx_D2 = (df["Smoothed_AtomicDensity"] - D2).abs().idxmin()
        X1 = df.loc[idx_D1, "Relative_x"]
        X2 = df.loc[idx_D2, "Relative_x"]

        # 准备结果
        result = pd.DataFrame({
            "filename": [os.path.basename(file_path)],
            "S1": [S1],
            "S_last": [S_last],
            "D1": [D1],
            "D2": [D2],
            "X1": [X1],
            "X2": [X2]
        })

        # 📤 输出到对应的文件夹 cal_d_9505.csv
        folder = os.path.dirname(file_path)
        output_file = os.path.join(folder, "cal_d_9010.csv")

        # 如果 cal_d_9505.csv 已存在，追加；否则新建
        if os.path.exists(output_file):
            result.to_csv(output_file, mode='a', index=False, header=False)
        else:
            result.to_csv(output_file, index=False)

        print(f"✅ 已处理：{file_path}")

    except Exception as e:
        print(f"❌ 处理 {file_path} 时出错：{e}")

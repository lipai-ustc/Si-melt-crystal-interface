import os

def change_xyz_type_to_si(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        line_num = 0
        for line in fin:
            # 第一行：原子数
            if line_num == 0:
                fout.write(line)
            # 第二行：注释行
            elif line_num == 1:
                fout.write(line)
            # 后面的行：原子数据
            else:
                parts = line.split()
                if len(parts) >= 4:
                    parts[0] = "Si"  # 替换成 Si
                    fout.write(" ".join(parts) + "\n")
                else:
                    fout.write(line)
            line_num += 1

def batch_convert_xyz(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".xyz"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(output_folder, filename)
            change_xyz_type_to_si(input_file, output_file)
            print(f"已处理: {filename} -> {output_file}")

if __name__ == "__main__":
    # 你可以修改下面两个路径
    input_folder = r"E:\free_energy\interface-varcha\111"     # 输入xyz文件夹路径
    output_folder = r"E:\free_energy\interface-varcha\111\result-post"   # 输出文件夹路径
    batch_convert_xyz(input_folder, output_folder)
    print("所有文件已处理完成！")

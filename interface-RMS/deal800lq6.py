def filter_even_frames(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    frame_lines = []
    frame_index = 0

    for line in lines:
        frame_lines.append(line)
        # 每一帧的第一行是原子数，第二行是注释，后面是原子坐标
        # 因此当frame_lines收集满一帧时，处理它
        if len(frame_lines) == int(frame_lines[0].strip()) + 2:
            if frame_index % 2 == 0:  # 只保留偶数帧
                filtered_lines.extend(frame_lines)
            frame_lines = []
            frame_index += 1

    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)


# 使用示例：
input_file = r"E:\free_energy\400-newpot\110\lq6.xyz"      # 你的原始文件名
output_file = r"E:\free_energy\400-newpot\110\lq6-2.xyz"   # 过滤后保存的新文件
filter_even_frames(input_file, output_file)

input_path = f"D:\中转站\fastsam-mindspore\weight\yolov8x-seg.txt"      # 输入文件路径
output_path = f"D:\中转站\fastsam-mindspore\weight\yolov8x-seg.txt"  # 输出文件路径

with open(input_path, "r") as f:
    lines = f.readlines()

# 删除每一行中第一个出现的 "model."
processed_lines = [line.replace("model.", "", 1) for line in lines]

# 写入新文件
with open(output_path, "w") as f:
    f.writelines(processed_lines)

print(f"处理完成，结果保存在：{output_path}")

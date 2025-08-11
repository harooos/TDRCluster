import csv
import random

def process_txt_to_csv(input_txt_path, output_csv_path, sample_size=1000):
    """
    从txt文件中随机抽取指定行数并生成CSV文件
    
    参数:
        input_txt_path: 输入txt文件路径
        output_csv_path: 输出csv文件路径
        sample_size: 要抽取的行数(默认1000)
    """
    # 读取原始文件所有行
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 去除每行开头的数字和可能的空格/制表符
    cleaned_lines = []
    for line in lines:
        # 去除行首数字和可能的空白字符
        cleaned_line = line.lstrip('0123456789').lstrip(' \t')
        cleaned_lines.append(cleaned_line.strip())
    
    # 随机抽取指定数量的行
    if len(cleaned_lines) > sample_size:
        sampled_lines = random.sample(cleaned_lines, sample_size)
    else:
        sampled_lines = cleaned_lines
        print(f"警告: 文件行数不足{sample_size}，将使用全部{len(cleaned_lines)}行")
    
    # 写入CSV文件
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for line in sampled_lines:
            writer.writerow([line])
    
    print(f"成功生成CSV文件: {output_csv_path}，包含{len(sampled_lines)}行数据")

if __name__ == "__main__":
    # 配置输入输出路径
    input_file = "D:/xuexi/python/LLM/Short-Text-Classification-main/VSQ/test.txt" # 替换为您的txt文件路径
    output_file = "D:/xuexi/python\LLM\TDRCluster/data/sample.csv"
    
    # 执行转换
    process_txt_to_csv(input_file, output_file)
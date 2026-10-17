import re
import json

def extract_performance_data(log_file):
    # 用于存储性能数据的字典
    performance_data = {}
    
    # 读取log文件
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 打印文件内容的前100个字符用于调试
    print("文件内容预览:", content[:100])
    
    # 使用正则表达式匹配所有性能数据
    # 修改匹配模式以更精确地匹配命令行参数和mykernel_time
    pattern = r'flash_llm_half_row_majorB_vs_bind_reorder_mw.*?-M (\d+) -N \d+ -K (\d+).*?mykernel_time: ([\d.]+) ms'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print("找到的匹配项:", matches)
    
    # 将数据按M和K分组
    data_by_size = {}
    for m, k, time in matches:
        key = f"{m}_{k}_0.9"
        if key not in data_by_size:
            data_by_size[key] = []
        data_by_size[key].append(float(time))
    
    print("分组后的数据:", data_by_size)
    
    # 组织数据格式
    for key, times in data_by_size.items():
        # 假设每个尺寸都有两个时间值，第一个是fused，第二个是not fused
        if len(times) == 2:
            performance_data[key] = {
                "fused": times[0],
                "not fused": times[1]
            }
    
    return performance_data

def main():
    log_file = "all_20250420-025641-fuse.log"
    output_file = "flash_llm_fusion_performance.json"
    
    # 提取数据
    performance_data = extract_performance_data(log_file)
    
    print("最终生成的JSON数据:", performance_data)
    
    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"数据已成功提取并保存到 {output_file}")

if __name__ == "__main__":
    main() 
import re
import json
import os

import statistics

def summarize(arr, name):
    if not arr:  # 避免空数组报错
        print(f"{name} 为空")
        return
    print(f"{name}: 平均数={statistics.mean(arr):.4f}, "
          f"中位数={statistics.median(arr):.4f}, "
          f"最大值={max(arr):.4f}, "
          f"最小值={min(arr):.4f}")

def load_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["filename"] = os.path.basename(item["filename"])
    result = {
        d["filename"]: {k: v for k, v in d.items() if k != "filename"}
        for d in data
    }
    return result


def parse_log(logfile, outfile):
    results = []
    current = None
    current_method = None

    duration_pattern = re.compile(r"Duration\s+(\w+)\s+([\d\.]+)")

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 新文件开始
            if line.startswith("正在测试"):
                if current:  # 存储上一个结果
                    results.append(current)
                filename = line.split("正在测试")[-1].strip().rstrip("...")
                current = {
                    "filename": filename,
                    "normalbinding": "",
                    "multibinding": "",
                    "strictlb": ""
                }
                current_method = None

            # 检测实现方法
            elif line.startswith("Normal Binding"):
                current_method = "normalbinding"
            elif line.startswith("Multi Binding"):
                current_method = "multibinding"
            elif line.startswith("Strict LB"):
                current_method = "strictlb"

            # 提取 duration
            elif "Duration" in line:
                m = duration_pattern.search(line)
                if m and current and current_method:
                    unit, value = m.groups()
                    value = float(value)
                    # 转换为毫秒
                    if unit.lower() == "us":
                        value = value / 1000.0
                    elif unit.lower() == "ms":
                        value = value
                    current[current_method] = f"{value:.5f}"
                    current_method = None  # 一旦拿到duration就重置，避免串行

        # 最后一个文件结果
        if current:
            results.append(current)

    strict_lb_fast_files = []
    multi_bind_fast_files = []
    # process results
    for record in results:
        if record["normalbinding"] == "" or record["multibinding"] == 0 or record["strictlb"] == 0:
            continue
        if record["multibinding"] < record["normalbinding"] and record["multibinding"] <= record["strictlb"]:
            multi_bind_fast_files.append(record["filename"])
        if record["strictlb"] < record["normalbinding"] and record["strictlb"] <= record["multibinding"]:
            strict_lb_fast_files.append(record["filename"])
        

    output = {
        "results": results,
        "strictlb_fastest": strict_lb_fast_files,
        "multi_bind_fastest": multi_bind_fast_files
    }
    with open(outfile, "w", encoding="utf-8") as f:
        # json.dump(results, f, indent=2, ensure_ascii=False)
        json.dump(output, f, indent=2, ensure_ascii=False)
    return strict_lb_fast_files, multi_bind_fast_files

if __name__ == "__main__":
    matrix_status_dict = load_from_json("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/matrix_status.json")
    strict_lb_fast_files, multi_bind_fast_files = parse_log("/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/test_all_mtx_lb.log", "/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/motivation_lb/results.json")

    strict_lb_files_std = [matrix_status_dict[os.path.basename(filepath.strip())]["nnz_std"] for filepath in strict_lb_fast_files]
    multi_bind_files_std = [matrix_status_dict[os.path.basename(filepath.strip())]["nnz_std"] for filepath in multi_bind_fast_files]


summarize(strict_lb_files_std, "Strict LB")
summarize(multi_bind_files_std, "Multi Binding")

    


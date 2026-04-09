import re

# 输入数据文件，每行一个plan
input_file = "/workspace/sparsene/examples/src_fp32/dtc/testbed/plans.txt"
output_file = "filtered_plans.txt"

pattern_num = re.compile(r"^(\d+),")        # 提取 plan 编号
pattern_x   = re.compile(r"\|\((\d+)\)>")   # 提取所有 x

valid_plans = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 提取编号
        m = pattern_num.match(line)
        if not m:
            continue
        plan_id = m.group(1)

        # 提取所有 (x)
        numbers = pattern_x.findall(line)

        # 如果存在并且全部是 "1"
        if numbers and all(n == "1" for n in numbers):
            valid_plans.append((plan_id, line))

# 输出 CSV，第一列是编号，第二列是plan内容
with open(output_file, "w", encoding="utf-8") as f:
    # f.write("plan_id,plan\n")
    for plan_id, plan in valid_plans:
        print(f"\"${{PROJECT_SOURCE_DIR}}/plans/plan_{plan_id}.inc\"")
        f.write(f"{plan}\n")


print(f"筛选完成，共找到 {len(valid_plans)} 个符合要求的 plan，已保存到 {output_file}")

import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

csv_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/draw_timeline/done-timeline-500row-window.1.csv"
csv_sm_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/draw_timeline/tb2sm.3.csv"
drawio_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/draw_timeline/timeline.5.drawio"

color_palette = ["#dae8fc", "#d5e8d4", "#fff2cc", "#f8cecc", "#e1d5e7", "#f5f5f5", "#e6efff", "#cce5ff"]
global_origin_start_min = float("inf")

def update_global_origin_start_min(origin_start):
    global global_origin_start_min
    global_origin_start_min = min(origin_start, global_origin_start_min)

def get_global_origin_start_min():
    global global_origin_start_min
    return global_origin_start_min

def parse_tb2sm_csv(csv_file):
    tb2sm = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            if row[0].startswith("block idx.x = "):
                m = re.match(r"block idx.x\s*=\s*(\d+)", row[0])
                if m:
                    current_block_x = int(m.group(1))
                    assert(len(tb2sm) == current_block_x)
                    # print(len(row), row[1:-1])
                    # assert(len(row) == 8 + 1)
                    tb2sm.append([int(x) for x in row[1:-1]])
    return tb2sm

def parse_csv_by_thread_block(csv_file):
    thread_blocks = defaultdict(list)

    current_block_x = None
    current_header = None

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 0:
                continue
            if row[0].startswith("block idx.x = "):
                # 解析新的 blockIdx.x
                m = re.match(r"block idx\.x\s*=\s*(\d+)", row[0])
                if m:
                    current_block_x = int(m.group(1))
                    current_header = None  # reset header
                    # print("DEBUG", "current_block_x", current_block_x)
                continue
            elif row[0] == "name":
                current_header = row  # 记录 header 行
                continue
            elif current_block_x is None or current_header is None:
                continue  # 没有 blockIdx.x 或 header 就跳过

            # 处理一个 op 行（逻辑与原来的类似）
            full_name = row[0]
            time_values = row[1:]

            # print("DEBUG", "full_name", full_name, "time_values", time_values)

            match_with_A2B = re.match(r"([A-Z]\d?[rs]?)2([A-Z]?\d?[rs]?)([A-Za-z]+)\((\d+)\)", full_name)
            match_without_A2B = re.match(r"([A-Za-z]+)\((\d+)\)", full_name)
            if not match_with_A2B and not match_without_A2B:
                continue
            if match_with_A2B:
                src, dst, op_name, index = match_with_A2B.group(1), match_with_A2B.group(2), match_with_A2B.group(3), int(match_with_A2B.group(4))
                dst = dst.capitalize()
                A2B = f"{src}2{dst}"
            elif match_without_A2B:
                src = "None"
                dst = "None"
                op_name, index = match_without_A2B.group(1), match_without_A2B.group(2)
                A2B = "None2None"

            # print("\tDEBUG", src, dst, op_name, index, A2B)
            for i in range(8):  # blockIdx.y = 0 ~ 7
                # start_idx = i * 4 + 2  # 跳过 origin start, origin end -> start, duration
                start_idx = i * 4
                try:
                    origin_start = int(time_values[start_idx])
                    origin_end = int(time_values[start_idx + 1])
                    start = float(time_values[start_idx + 2])
                    duration = float(time_values[start_idx + 3])
                    block_y = i
                    update_global_origin_start_min(origin_start)
                    thread_blocks[(current_block_x, block_y)].append({
                        "full_name": full_name,
                        "A2B": A2B,
                        "op_name": op_name,
                        "index": index,
                        "origin_start": origin_start,
                        "origin_end": origin_end,
                        "start": start,
                        "duration": duration
                    })
                except:
                    continue
    # print("DEBUG", "thread blocks", thread_blocks)
    return thread_blocks

# ---------- 画图核心 ----------
def create_cell(text, x, y, width, height, cell_id, color="#dae8fc", dashed=False, fill=True):
    style = f"shape=rectangle;strokeColor=#000000;fontSize=10;"
    if dashed:
        style += "dashed=1;"
    if fill:
        style += f"fillColor={color};"
    else:
        style += "fillColor=none;"
    cell = ET.Element("mxCell", {
        "id": str(cell_id),
        "value": text,
        "style": style,
        "vertex": "1",
        "parent": "1"
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x),
        "y": str(y),
        "width": str(width),
        "height": str(height),
        "as": "geometry"
    })
    return cell

# ---------- thread block 局部绘图 ----------
def thread_block_draw(x_offset, y_offset, ops, block_id_x, block_id_y, cell_id_start=1000):
    op_y_map = {}
    op_order = []
    height_per_op = 30
    cell_list = []

    # 分配 op_name 对应的 y 相对位置
    for op in ops:
        key = op["op_name"]
        if key not in op_y_map:
            op_y_map[key] = len(op_y_map) * height_per_op
            op_order.append(key)

    color_map = {}
    for i, op_name in enumerate(op_y_map.keys()):
        color_map[op_name] = color_palette[i % len(color_palette)]

    cid = cell_id_start
    for op in ops:
        rel_y = op_y_map[op["op_name"]]
        x = x_offset + op["start"]
        y = y_offset + rel_y
        width = op["duration"]
        height = height_per_op

        # label = f"{op['A2B']}\n{op['op_name']}\n{round(op['start'],1)}~{round(op['start']+op['duration'],1)}"
        label = f"{op['A2B']}\n{op['op_name']}\n{op['index']}"
        color = color_map[op['op_name']]
        cell = create_cell(label, x, y, width, height, cid, color)
        cell_list.append(cell)
        cid += 1

    # 返回该 block 的绘图高度
    total_height = len(op_y_map) * height_per_op
    max_x = max(op['start'] + op['duration'] for op in ops)

    # ---------- 添加虚线边框 ----------
    padding = 10
    block_x0 = x_offset + min(op['start'] for op in ops) - padding
    block_x1 = x_offset + max(op['start'] + op['duration'] for op in ops) + padding
    block_y0 = y_offset
    block_y1 = y_offset + total_height

    block_width = block_x1 - block_x0
    block_height = block_y1 - block_y0

    label = f"TB ({block_id_x},{block_id_y})"
    border_cell = create_cell(
        text="",
        x=block_x0,
        y=block_y0,
        width=block_width,
        height=block_height,
        cell_id=cid,
        color=None,
        dashed=True,  # 使其变成虚线边框
        fill=False
    )
    cell_list.append(border_cell)
    cid += 1

    # ---------- 添加一个label ----------
    label_width = 60
    label_height = 20
    label_x = block_x0 + block_width - label_width
    label_y = block_y0
    label_box = create_cell(
        text=label,
        x=label_x,
        y=label_y,
        width=label_width,
        height=label_height,
        cell_id=cid,
        color="#DFC6C6",
    )
    cell_list.append(label_box)
    cid += 1


    return cell_list, cid, total_height, max_x

# ---------- 调度器 ----------
def layout_thread_blocks(thread_blocks, tb2sm):
    block_positions = []  # (block_id_x, block_id_y, x, y, width, height)
    placed_blocks = []

    for (block_id_x, block_id_y), ops in sorted(thread_blocks.items()):
        if tb2sm[block_id_x][block_id_y] != 0:
            continue
        # print(f"DEBUG Thread Block: blockIdx.x={block_id_x}, blockIdx.y={block_id_y}")

        # 计算 block 的 width, height
        max_x = max(op['start'] + op['duration'] for op in ops)
        
        width = max_x
        height_per_op = 30
        num_ops = len(set(op['op_name'] for op in ops))
        height = num_ops * height_per_op

        # 放置位置调度：尝试从 y = 0 开始排，如果 x 冲突就往下挪
        thread_block_origin_start_min = min(op['origin_start'] for op in ops)
        x = (thread_block_origin_start_min - get_global_origin_start_min()) / 10
        y = 0
        while True:
            conflict = False
            for _, _, px, py, pw, ph in placed_blocks:
                if py == y and not (x + width <= px or x >= px + pw):
                    y = py + ph + 20  # 向下挪
                    conflict = True
                    break
            if not conflict:
                break

        block_positions.append((block_id_x, block_id_y, x, y, width, height))
        placed_blocks.append((block_id_x, block_id_y, x, y, width, height))
    return block_positions

# ---------- Drawio XML 生成 ----------
def generate_drawio(thread_blocks, output_file, tb2sm):
    root = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(root, "diagram", name="Blocks")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1000", dy="1000", grid="1", gridSize="10", guides="1")
    rootCell = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(rootCell, "mxCell", {"id": "0"})
    ET.SubElement(rootCell, "mxCell", {"id": "1", "parent": "0"})

    positions = layout_thread_blocks(thread_blocks, tb2sm)
    # print("DEBUG", positions)
    cid = 1000

    for block_id_x, block_id_y, x, y, w, h in positions:
        # print("DEBUG", "block_id_x", block_id_x, "block_id_y", block_id_y)
        ops = thread_blocks[(block_id_x, block_id_y)]
        cells, cid, _, _ = thread_block_draw(x, y, ops, block_id_x, block_id_y, cid)
        for cell in cells:
            rootCell.append(cell)

    ET.ElementTree(root).write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"✅ Draw.io 文件保存至: {output_file}")

# ---------- 主函数 ----------
if __name__ == "__main__":
    tb2sm = parse_tb2sm_csv(csv_sm_file)
    tb_ops = parse_csv_by_thread_block(csv_file)
    # print(1)
    # for (block_x, block_y), ops in tb_ops.items():
    #     if tb2sm[block_x][block_y] != 0:
    #         continue
    #     print(f"Thread Block: blockIdx.x={block_x}, blockIdx.y={block_y}")
    #     for op in ops:
    #         print(f"  Op {op['full_name']}: start={op['start']}, duration={op['duration']}")
    # print(2)
    generate_drawio(tb_ops, drawio_file, tb2sm)
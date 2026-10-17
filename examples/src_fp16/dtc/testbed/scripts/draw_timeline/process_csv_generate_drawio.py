import csv
import xml.etree.ElementTree as ET

# CSV 文件路径
csv_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/timeline-2row-window.csv"
drawio_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/timeline.drawio"

# 读取 CSV
def parse_csv(csv_file):
    ops = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            name = row[0]
            time_values = row[1:]

            # 解析 op name: e.g., G2rSparseOffsetLoad(0)
            import re
            m = re.match(r"([A-Z]\d?[rs]?)2([A-Z]\d?[rs]?)?([A-Za-z]+)\((\d+)\)", name)
            if not m:
                if name.startswith("calculate"):
                    src, dst, optype, idx = "CALC", "CALC", "calculate", int(name.split("(")[-1].split(")")[0])
                else:
                    continue
            else:
                src, dst, optype, idx = m.group(1), m.group(2), m.group(3), int(m.group(4))

            # 处理 8 组时间数据，每组4列：origin_start, origin_end, start, duration
            time_data = []
            for i in range(0, len(time_values), 4):
                try:
                    start = float(time_values[i+2])
                    duration = float(time_values[i+3])
                    time_data.append((start, duration))
                except:
                    continue

            ops.append({
                "src": src,
                "dst": dst,
                "optype": optype,
                "index": idx,
                "timeline": time_data,
                "name": name
            })
    return ops

# 生成 drawio XML
def generate_drawio(ops, output_file):
    cell_id = 1
    op_height = 30
    op_spacing = 10

    def create_cell(opname, x, y, width, height, cell_id, parent="1"):
        cell = ET.Element("mxCell", {
            "id": str(cell_id),
            "value": opname,
            "style": "shape=rect;fillColor=#dae8fc;strokeColor=#6c8ebf;",
            "vertex": "1",
            "parent": parent
        })
        geometry = ET.SubElement(cell, "mxGeometry", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "as": "geometry"
        })
        return cell

    # 建立 XML 结构
    root = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(root, "diagram", name="Timeline")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1000", dy="1000", grid="1", gridSize="10", guides="1")
    rootCell = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(rootCell, "mxCell", {"id": "0"})
    ET.SubElement(rootCell, "mxCell", {"id": "1", "parent": "0"})

    for i, op in enumerate(ops):
        y = i * (op_height + op_spacing)
        for j, (start, duration) in enumerate(op["timeline"]):
            x = start
            width = duration
            label = f"{op['src']}2{op['dst']}:{op['optype']}({op['index']})[{j}]"
            cell = create_cell(label, x, y, width, op_height, cell_id + 1)
            rootCell.append(cell)
            cell_id += 1

    # 保存 XML
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# 主函数
if __name__ == "__main__":
    ops = parse_csv(csv_file)
    generate_drawio(ops, drawio_file)
    print(f"Drawio file saved to {drawio_file}")

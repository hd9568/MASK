import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

csv_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/draw_timeline/timeline-500row-window.1.csv"
csv_out_file = "/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/draw_timeline/done-timeline-500row-window.1.csv"

def add_op_names_for_csv_lines(csv_lines_to_process):
    if len(csv_lines_to_process) == 0:
        return []
    if len(csv_lines_to_process) not in range(1, 100, 7):
        print("NOT CORRECT")
    kernel_len = int((len(csv_lines_to_process) - 1) / 7)
    csv_lines_to_process[0][0] = "G2rSparseoffsetLoad(0)"
    # print("G2rSparseOffsetLoad(0)")
    current_ptr = 1
    for i in range(kernel_len + 3):
        if i < kernel_len:
            # print(f"G2sSparseIndexLoad({i})")
            # print(f"G2sSparseCooEncLoad({i})")
            # print(f"G2sMatrixBLoad({i})")
            # print(f"S2sRestoreMatrixA({i})")
            csv_lines_to_process[current_ptr][0] = f"G2sSparseIndexLoad({i})"
            csv_lines_to_process[current_ptr + 1][0] = f"G2sSparseCooEncLoad({i})"
            csv_lines_to_process[current_ptr + 2][0] = f"G2sMatrixBLoad({i})"
            csv_lines_to_process[current_ptr + 3][0] = f"S2sRestoreMatrixA({i})"
            current_ptr = current_ptr + 4
        if i - 1 >= 0 and i - 1 < kernel_len:
            # print(f"S2rAValLoad({i - 1})")
            # print(f"S2rBValLoad({i - 1})")
            csv_lines_to_process[current_ptr][0] = f"S2rAValLoad({i - 1})"
            csv_lines_to_process[current_ptr + 1][0] = f"S2rBValLoad({i - 1})"
            current_ptr = current_ptr + 2
        if i - 2 >= 0 and i - 2 < kernel_len:
            # print(f"calculate({i - 2})")
            csv_lines_to_process[current_ptr][0] = f"calculate({i - 2})"
            current_ptr = current_ptr + 1
    return csv_lines_to_process

def parse_csv_by_thread_block(csv_file):
    thread_blocks = defaultdict(list)
    
    current_block_x = None
    current_header = None

    csv_lines = []

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        csv_lines_to_process = []
        for row in reader:
            if len(row) == 0:
                continue
            if row[0].startswith("block idx.x = "):
                m = re.match(r"block idx.x\s*=\s*(\d+)", row[0])
                if m: 
                
                    # process last csv file
                    csv_lines_to_process = add_op_names_for_csv_lines(csv_lines_to_process)
                    csv_lines.extend(csv_lines_to_process)
                    csv_lines_to_process = []

                    # process current line
                    current_block_x = int(m.group(1))
                    current_header = None
                    csv_lines.append(row)
                    print("DEBUG", "current_block_x", current_block_x)
                continue
            elif row[0] == "name":
                current_header = row
                csv_lines.append(row)
                continue
            elif current_block_x is None or current_header is None:
                continue
            elif row[1] == "":
                # csv_lines.append(row)
                continue

            
            full_name = row[0]
            if full_name == "":
                csv_lines_to_process.append(row)
            else:
                csv_lines.append(row)
            # print(row[0], row[0] == "")
        csv_lines_to_process = add_op_names_for_csv_lines(csv_lines_to_process)
        csv_lines.extend(csv_lines_to_process)
        csv_lines_to_process = []
    return csv_lines

if __name__ == "__main__":
    csv_lines = parse_csv_by_thread_block(csv_file)
    with open(csv_out_file, mode="w", newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(csv_lines)
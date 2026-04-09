import os

def collect_mtx_files(root_dir, output_file):
    mtx_files = []
    # 遍历文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mtx"):
                full_path = os.path.join(dirpath, filename)
                mtx_files.append(full_path)

    # 写入到 txt 文件
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in mtx_files:
            f.write(file_path + "\n")

    print(f"共找到 {len(mtx_files)} 个 .mtx 文件，已写入 {output_file}")

if __name__ == "__main__":
    root_directory = "/workspace/selectedMM/"  # 修改为你要遍历的文件夹路径
    output_txt = "/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/mtx_files.txt"
    collect_mtx_files(root_directory, output_txt)

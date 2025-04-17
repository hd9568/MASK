import pathlib
import argparse
import json
from datetime import datetime

script_path = pathlib.Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-o", type=str, default=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    parser.add_argument("-g", type=str, default="H800")
    args = parser.parse_args()

    data = {}

    log_list = args.i
    output_path = args.o

    has_filename = False
    has_exe_name = False
    has_n = False
    has_duration = False

    with open(log_list, "r") as f:
        log_paths = [line.strip() for line in f]
        for log_path in log_paths:
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Filename: "):
                        filename = line.split(" ")[1].strip()
                        mat_name = filename.split("/")[-1]
                        has_filename = True
                    elif "-filename" in line:
                        exe_name = line.split(" ")[0].strip().split("/")[-1]
                        demo_name = "-".join(exe_name.split("-")[:-1])
                        if demo_name.startswith("demo-"):
                            demo_name = demo_name[5:]
                        prec = exe_name.split("-")[-1]
                        has_exe_name = True
                    elif line.startswith("N:"):
                        n = int(line.split(" ")[1].strip())
                        has_n = True
                    elif line.startswith("mykernel_time:"):
                        duration = float(line.split(" ")[1].strip())
                        has_duration = True
                    elif "=" * 100 in line:
                        if has_filename and has_exe_name and has_n and has_duration:
                            data.setdefault(args.g, {}).setdefault(prec, {}).setdefault(
                                mat_name, {}
                            ).setdefault(n, {}).setdefault(demo_name, []).append(
                                duration
                            )
                        has_filename = False
                        has_exe_name = False
                        has_n = False
                        has_duration = False

    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

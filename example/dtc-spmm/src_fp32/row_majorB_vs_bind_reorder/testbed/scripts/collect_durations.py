import os
from pathlib import Path
import subprocess

ncu_report_dir = Path(__file__).parent / "ncu_reports"

with open(Path(__file__).parent / "durations.csv", "w") as f:
    f.write("file_name,duration,unit\n")
    for ncu_report_path in ncu_report_dir.glob("*.ncu-rep"):
        file_name = ncu_report_path.stem
        print(file_name)
        result = subprocess.run(
            f"ncu --import {ncu_report_path} --csv | grep Duration | grep 'cute'",
            shell=True,
            capture_output=True,
            text=True
        )
        columns = result.stdout.split(",")
        duration = columns[41]
        duration_unit = columns[40]
        f.write(f"{file_name},{duration},{duration_unit}\n")
        f.flush()

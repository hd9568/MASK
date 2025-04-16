import os
from pathlib import Path
import subprocess
import csv
from io import StringIO

ncu_report_dir = Path(__file__).parent / "ncu_reports"

with open(Path(__file__).parent / "durations.csv", "w") as f, open(Path(__file__).parent / "full.csv", "w") as f_full:
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
        # Parse the CSV output properly
        csv_reader = csv.reader(StringIO(result.stdout))
        for row in csv_reader:
            if row:  # Skip empty rows
                # print(row)
                duration = row[14]
                duration_unit = row[13]
                f.write(f"{file_name},{duration},{duration_unit}\n")
                f_full.write(result.stdout)
                f.flush()
                f_full.flush()
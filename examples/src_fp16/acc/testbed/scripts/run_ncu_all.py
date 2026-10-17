from pathlib import Path
import os
import subprocess

script_path = Path(__file__).parent
build_path = script_path / ".." / "build"
ncu_report_path = script_path / "ncu_reports"
ncu_report_path.mkdir(exist_ok=True)

# Create directory for logs if it doesn't exist
log_path = script_path / "logs"
log_path.mkdir(exist_ok=True)

for file in build_path.glob("plan_*"):
    filename = file.name
    if "." in filename:
        continue

    # Run the command and capture output
    result = subprocess.run(
        [
            "ncu",
            "-f",
            "-o",
            str(ncu_report_path / filename),
            "--import-source",
            "on",
            "--set",
            "full",
            str(file),
            "-mtx_flag",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    # Save stdout and stderr to files
    with open(log_path / f"{filename}_stdout.log", "w") as f:
        f.write(result.stdout)
    with open(log_path / f"{filename}_stderr.log", "w") as f:
        f.write(result.stderr)

    print(filename)

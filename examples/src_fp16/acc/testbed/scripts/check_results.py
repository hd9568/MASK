from pathlib import Path

logs_dir = Path(__file__).parent / "logs"

for log_path in logs_dir.glob("*.log"):
    file_name = log_path.stem
    with open(log_path, "r") as f:
        lines = f.readlines()
        if len(lines) > 14:
            print(file_name, len(lines))

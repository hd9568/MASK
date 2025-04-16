import os
import pathlib
import subprocess
from tqdm import tqdm
from datetime import datetime
import sys

N = [64, 128, 256, 512]

script_path = pathlib.Path(__file__).parent
mat_list_path = script_path / "mat_list.txt"
prog_list_path = script_path / "prog_list.txt"

with open(mat_list_path, "r") as f:
    mats = [line.strip() for line in f]

with open(prog_list_path, "r") as f:
    progs = [line.strip() for line in f]

print(mats)
print(progs)

log_path = script_path / f"all_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
with open(log_path, "w") as f:
    # Create progress bars outside the loops
    mat_bar = tqdm(mats, desc="Matrix", position=1, leave=False)
    for mat in mat_bar:
        prog_bar = tqdm(progs, desc="Program", position=2, leave=False)
        for prog in prog_bar:
            n_bar = tqdm(N, desc="N Size", position=0)
            for n in n_bar:
                # Get current date
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Run the main command
                shell_cmd = f"{prog} -filename {mat} -N {n}"
                f.write(shell_cmd + "\n")
                f.flush()  # Ensure command is written to file immediately
                
                process = subprocess.Popen(
                    shell_cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # Enable text mode
                )
                
                # Stream output in real-time
                while True:
                    output = process.stdout.readline()
                    error = process.stderr.readline()
                    
                    if output == "" and error == "" and process.poll() is not None:
                        break
                    
                    if output:
                        print(output.strip())  # Print to console
                        f.write(output)  # Write to file
                        f.flush()  # Ensure immediate file write
                    if error:
                        print(error.strip(), file=sys.stderr)  # Print to console
                        f.write(error)  # Write to file
                        f.flush()  # Ensure immediate file write

                f.write("=" * 100 + "\n")
                f.flush()  # Ensure separator is written to file
                
                # Update all progress bars
                n_bar.refresh()
                prog_bar.refresh()
                mat_bar.refresh()

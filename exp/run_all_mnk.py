import os
import pathlib
import subprocess
from tqdm import tqdm
from datetime import datetime
import sys
import argparse

script_path = pathlib.Path(__file__).parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prog-list", type=str, default=str(script_path / "prog_list.txt")
    )
    args = parser.parse_args()

    MNK = [
        (512, 512, 512),
        (1024, 512, 1024),
        (2048, 512, 2048),
        (4096, 512, 4096),
    ]

    prog_list_path = args.prog_list

    with open(prog_list_path, "r") as f:
        progs = [line.strip() for line in f]

    print(progs)

    log_path = script_path / f"all_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    with open(log_path, "w") as f:
        # Create progress bars outside the loops
        prog_bar = tqdm(progs, desc="Program", position=2, leave=False)
        for prog in prog_bar:
            mnk_bar = tqdm(MNK, desc="MNK", position=0)
            for mnk in mnk_bar:
                # Get current date
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Run the main command
                shell_cmd = f"{prog} -mtx_flag 0 -M {mnk[0]} -N {mnk[1]} -K {mnk[2]}"
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
                mnk_bar.refresh()
                prog_bar.refresh()

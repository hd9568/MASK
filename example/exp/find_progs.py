import pathlib
import argparse


def find_progs(path: pathlib.Path):
    for file in path.glob("**/*-fp16"):
        print(file.absolute())
    for file in path.glob("**/*-tf32"):
        print(file.absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the program directory")
    args = parser.parse_args()
    find_progs(pathlib.Path(args.path))

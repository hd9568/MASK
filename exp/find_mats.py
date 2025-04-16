import pathlib
import argparse


def find_mats(path: pathlib.Path):
    for file in path.glob("**/*.mtx"):
        print(file.absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the matrix market directory")
    args = parser.parse_args()
    find_mats(pathlib.Path(args.path))

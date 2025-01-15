import re


def indent(s: str, width: int = 4):
    return re.sub(r'^', ' ' * width, s, flags=re.M)

import re


def indent(s: str, width: int = 4, char: str = ' '):
    return re.sub(r'^', char * width, s, flags=re.M)

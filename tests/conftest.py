import sys

collect_ignore: list[str] = []
if sys.platform.startswith("win"):
    collect_ignore.append("tests/maxsat")

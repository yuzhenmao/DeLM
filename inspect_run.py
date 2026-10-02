"""Print a saved run's measurements and board without any model call."""

import argparse
from pathlib import Path

from reporting.report import run_report as report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    print(report(parser.parse_args().folder), end="")

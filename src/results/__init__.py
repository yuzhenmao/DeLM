"""Results persistence ().

Trajectory JSON + results.csv writers extracted out of the runner so the
solver loop is purely orchestration; persistence is a separate concern.
"""
from src.results.csv_writer import save_csv_row
from src.results.trajectory import save_trajectory

__all__ = ["save_trajectory", "save_csv_row"]

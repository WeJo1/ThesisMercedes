import csv
from datetime import datetime
from pathlib import Path
from typing import Dict


def log(message: str) -> None:
    """Print a timestamped message to the console."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


class CSVLogger:
    """Append scalar metrics to a CSV file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log_row(self, row: Dict[str, object]) -> None:
        fields = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

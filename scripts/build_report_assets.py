"""Validate tracked result tables and regenerate SVG report charts."""

from pathlib import Path

from src.reporting import write_report_assets


if __name__ == "__main__":
    paths = write_report_assets(Path("reports"))
    for path in paths:
        print(f"saved: {path}")

from pathlib import Path
from datetime import datetime
import runpy
import traceback

ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "run_status.txt"

SCRIPT = ROOT / "scripts" / "ict" / "63_fvg_scaleout_by_year_max_stop_worst_losses.py"


def write_status(status, message):
    STATUS.write_text(
        f"status={status}\n"
        f"timestamp={datetime.now()}\n"
        f"message={message}\n",
        encoding="utf-8",
    )

try:
    print("LATEST RESEARCH: FVG BY YEAR / MAX STOP / WORST LOSSES")
    print(f"Running delegated script: {SCRIPT}")
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Missing delegated script: {SCRIPT}")
    runpy.run_path(str(SCRIPT), run_name="__main__")
    write_status("SUCCESS", "Completed FVG by-year max-stop worst-loss study via latest_research.py")
except Exception as exc:
    print(traceback.format_exc())
    write_status("FAILED", f"{type(exc).__name__}: {exc}")
    raise

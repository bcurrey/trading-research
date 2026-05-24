from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"

OUT.mkdir(parents=True, exist_ok=True)

print("Runner test successful")
print(datetime.now())

(OUT / "runner_test.txt").write_text(
    f"Runner worked at {datetime.now()}",
    encoding="utf-8"
)

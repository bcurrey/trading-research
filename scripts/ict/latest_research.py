from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"

OUT.mkdir(parents=True, exist_ok=True)

now = datetime.now()

print("AUTO TRIGGER TEST")
print(now)

(OUT / "runner_test.txt").write_text(
    f"Auto trigger worked at {now}",
    encoding="utf-8"
)

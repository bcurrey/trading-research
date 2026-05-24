from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

now = datetime.now()
message = f"Quick pipeline test succeeded at {now}\n"

print("QUICK PIPELINE TEST")
print(message)

(OUT / "quick_pipeline_test.txt").write_text(message, encoding="utf-8")

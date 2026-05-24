from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"

OUT.mkdir(parents=True, exist_ok=True)

now = datetime.now()

print("CHATGPT DIRECT WRITE TEST SUCCESS")
print(now)

(OUT / "chatgpt_write_test.txt").write_text(
    f"ChatGPT direct write succeeded at {now}\n",
    encoding="utf-8"
)

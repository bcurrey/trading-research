from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")

print("Raw data folder:", RAW_DIR)
print("\nFiles found:")

files = sorted([f for f in RAW_DIR.iterdir() if f.is_file()])

if not files:
    print("No files found.")
else:
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"{file.name} | {size_mb:,.2f} MB")
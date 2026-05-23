import zipfile
from pathlib import Path
from datetime import datetime
import polars as pl
import zstandard as zstd
import tempfile

RAW_ZIP = Path(r"D:\TradingResearch\data_raw\GLBX-20260520-PPUWMN7U6T.zip")
OUT_DIR = Path(r"D:\TradingResearch\data_parquet\tick_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "NQ_trades_2026_02_03_04.parquet"

print("\nIMPORT NQ TRADES TICK DATA")
print("Started:", datetime.now())
print("ZIP:", RAW_ZIP)

wanted_cols = ["ts_event", "price", "size", "side", "symbol"]
frames = []

with zipfile.ZipFile(RAW_ZIP, "r") as z:
    files = [name for name in z.namelist() if name.endswith(".trades.csv.zst")]

    print("Trade files found:", len(files))

    for i, name in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Reading {name}")

        with z.open(name) as compressed:
            dctx = zstd.ZstdDecompressor()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp_path = Path(tmp.name)
                dctx.copy_stream(compressed, tmp)

        try:
            day_df = pl.read_csv(
                tmp_path,
                columns=wanted_cols,
                try_parse_dates=True,
                infer_schema_length=1000,
            )

            day_df = day_df.filter(pl.col("symbol").str.starts_with("NQ"))

            day_df = day_df.with_columns([
                pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC")),
                pl.col("price").cast(pl.Float64),
                pl.col("size").cast(pl.Int64),
                pl.col("side").cast(pl.Utf8),

                pl.when(pl.col("side") == "A")
                .then(pl.col("size"))
                .when(pl.col("side") == "B")
                .then(-pl.col("size"))
                .otherwise(0)
                .alias("signed_size"),
            ])

            frames.append(day_df)

        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

if not frames:
    raise SystemExit("No trade files loaded.")

all_trades = pl.concat(frames).sort("ts_event")

print("\nRows loaded:", all_trades.height)

print("\nDate range:")
print(
    all_trades.select([
        pl.col("ts_event").min().alias("first_ts"),
        pl.col("ts_event").max().alias("last_ts"),
        pl.len().alias("rows"),
    ])
)

print("\nSide counts:")
print(
    all_trades.group_by("side")
    .agg([
        pl.len().alias("trades"),
        pl.col("size").sum().alias("contracts"),
    ])
    .sort("trades", descending=True)
)

print("\nWriting:")
print(OUT_FILE)

all_trades.write_parquet(OUT_FILE, compression="zstd")

print("\nDONE.")
print("Finished:", datetime.now())
"""
Look at the raw real-world datasets in data/raw/ before writing loaders.
Prints, per file: shape, columns + types, first rows, missing values, time range.
Run: python inspect_data.py
"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
FILES = {
    "Bike Sharing (hourly)": "hour.csv",
    "Beijing PM2.5": "PRSA_data_2010.1.1-2014.12.31.csv",
    "Appliances Energy": "energydata_complete.csv",
}

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

for title, fname in FILES.items():
    path = RAW / fname
    print("=" * 100)
    print(f"{title}   ({path})")
    if not path.exists():
        print("  MISSING FILE")
        continue
    df = pd.read_csv(path)
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("columns (type, distinct values, example):")
    for c in df.columns:
        print(f"  {c:12s} {str(df[c].dtype):8s} distinct={df[c].nunique():6d}  e.g. {df[c].iloc[0]!r}")

    print("\nfirst 5 rows:")
    print(df.head().to_string())

    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("\nmissing values:", "none" if miss.empty else "")
    for c, n in miss.items():
        print(f"  {c}: {n} ({100 * n / len(df):.1f}%)")

    # Time column(s): each file stores time differently.
    print("\ntime:")
    if "date" in df.columns:                                   # Appliances: one datetime string
        t = pd.to_datetime(df["date"])
        print("  column 'date' (datetime string)")
    elif "dteday" in df.columns:                               # Bike: date + hour
        t = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
        print("  columns 'dteday' + 'hr'")
    else:                                                      # PM2.5: year/month/day/hour
        t = pd.to_datetime(df[["year", "month", "day", "hour"]])
        print("  columns 'year','month','day','hour'")
    step = t.diff().dropna()
    print(f"  range: {t.min()}  ->  {t.max()}")
    print(f"  already sorted by time: {t.is_monotonic_increasing}   duplicate times: {t.duplicated().sum()}")
    print(f"  most common step: {step.mode().iloc[0]}   gaps larger than that: {(step > step.mode().iloc[0]).sum()}")
    print()

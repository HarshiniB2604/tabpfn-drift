"""
Real-world regression streams, built from the raw CSVs in data/raw/ (not committed).

Each loader returns a RealStream with:
- df           : one row per time step, strictly in time order (never shuffled), with
                 the feature columns, the target `y`, the `block` number (200 rows each,
                 leftover rows at the end dropped) and a `time` column (for plots only).
- feature_cols : which columns are model inputs (names differ per dataset; the runner
                 takes this list instead of assuming f0..f9).
- description  : one short sentence for reports and the app.

No future information: every feature is measured at the same time step as its target,
nothing is computed from later rows, and no statistics of the whole file are used.
There is no known drift point in real data (drift_blocks_for returns []).
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from common.contract import BLOCK_SIZE

RAW = Path(__file__).resolve().parent / "raw"


@dataclass
class RealStream:
    name: str
    df: pd.DataFrame
    feature_cols: list
    description: str


def _finish(name, df, time, target, feature_cols, description, block_size=BLOCK_SIZE):
    """Order by time, keep features + y, cut into blocks of block_size rows."""
    out = df[feature_cols].astype(float).copy()
    out["y"] = df[target].astype(float)
    out["time"] = time
    out = out.sort_values("time", kind="stable").reset_index(drop=True)   # files are already sorted
    n_blocks = len(out) // block_size
    out = out.iloc[: n_blocks * block_size].copy()                       # drop the unfinished last block
    out["block"] = out.index // block_size
    assert out[feature_cols + ["y"]].notna().all().all(), "missing values left"
    assert out["time"].is_monotonic_increasing
    return RealStream(name, out, list(feature_cols), description)


def load_bike(block_size=BLOCK_SIZE):
    df = pd.read_csv(RAW / "hour.csv")
    time = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    # Dropped: instant (row id), dteday (date), yr (just "year 1 or 2", a time label),
    # casual + registered (they add up to the target cnt -> would leak the answer).
    features = ["season", "mnth", "hr", "holiday", "weekday", "workingday",
                "weathersit", "temp", "atemp", "hum", "windspeed"]
    return _finish("bike", df, time, "cnt", features,
                   "Bike Sharing (Washington D.C., 2011-2012): hourly bike rentals from "
                   "calendar and weather.", block_size)


def load_pm25(block_size=BLOCK_SIZE):
    df = pd.read_csv(RAW / "PRSA_data_2010.1.1-2014.12.31.csv")
    df = df[df["pm2.5"].notna()].copy()        # target missing -> drop row (never invent labels)
    time = pd.to_datetime(df[["year", "month", "day", "hour"]])
    # Wind direction is text: one 0/1 column per direction (fixed list, not learned from the file).
    for d in ("NE", "NW", "SE", "cv"):
        df[f"cbwd_{d}"] = (df["cbwd"] == d).astype(float)
    # Dropped: No (row id), year + day (time labels), cbwd (replaced by the 0/1 columns).
    features = ["month", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
                "cbwd_NE", "cbwd_NW", "cbwd_SE", "cbwd_cv"]
    return _finish("pm25", df, time, "pm2.5", features,
                   "Beijing PM2.5 (US Embassy, 2010-2014): hourly air pollution from weather; "
                   "rows with missing PM2.5 removed.", block_size)


def load_energy(block_size=BLOCK_SIZE):
    df = pd.read_csv(RAW / "energydata_complete.csv")
    time = pd.to_datetime(df["date"])
    df["hour_of_day"] = time.dt.hour           # known at that moment, no future information
    # Dropped: date (time), rv1 + rv2 (random noise columns added by the dataset authors).
    rooms = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]
    weather = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    features = ["lights"] + rooms + weather + ["hour_of_day"]
    return _finish("energy", df, time, "Appliances", features,
                   "Appliances Energy (one house in Belgium, Jan-May 2016): appliance energy use "
                   "every 10 minutes from room and weather sensors.", block_size)


LOADERS = {"bike": load_bike, "pm25": load_pm25, "energy": load_energy}


def load_real(name, block_size=BLOCK_SIZE):
    return LOADERS[name](block_size)

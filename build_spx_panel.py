"""
build_spx_panel.py
==================

Build one clean DAILY table for the S&P 500 index (OptionMetrics secid 108105):
implied-volatility surface features, realized-variance features, a risk-free
rate, VIX, and a 21-day-ahead realized-variance target.

The script runs in three steps, and you can run each one by itself:

    python build_spx_panel.py --list-tables   # step 0: print WRDS libraries/tables, then stop
    python build_spx_panel.py --download      # step 1: WRDS + FRED -> raw files on disk
    python build_spx_panel.py --build         # step 2: raw files on disk -> panel + checks
    python build_spx_panel.py                 # all three in sequence

Step 2 never contacts WRDS. It reads only the raw files written by step 1, so the
build can be reproduced from a frozen snapshot.

Credentials: wrds.Connection() asks for your WRDS username and password itself.
Nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
SECID = 108105                      # OptionMetrics security id of the S&P 500 index (SPX)

# OptionMetrics libraries to search, in order of preference. 'optionm' usually holds
# year-split tables (vsurfd1996, ..., vsurfd2024); 'optionm_all' may hold stacked ones.
LIBRARY_CANDIDATES = ["optionm", "optionm_all"]

# Table "families". A family is either year-split (prefix + YYYY) or a single table.
SURFACE_PREFIX = "vsurfd"           # volatility surface
PRICE_PREFIX = "secprd"             # security prices
ZERO_PREFIX = "zerocd"              # zero-coupon curve

# Columns to pull. Required columns must exist. Optional ones are pulled if present.
SURFACE_COLS_REQUIRED = ["secid", "date", "days", "delta", "cp_flag", "impl_volatility"]
SURFACE_COLS_OPTIONAL = ["impl_strike"]
PRICE_COLS_REQUIRED = ["secid", "date", "close"]
ZERO_COLS_REQUIRED = ["date", "days", "rate"]

# Sample. None means "everything available": earliest to latest date in the data.
START_DATE: str | None = None       # e.g. "1996-01-01"
END_DATE: str | None = None         # e.g. "2024-12-31"

# FRED VIX (VIXCLS, in index points).
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

# File locations (relative to this script unless you pass --data-dir).
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
FILE_FORMAT = "parquet"             # "parquet" (needs pyarrow) or "csv"
RAW_DATE_TAG: str | None = None     # e.g. "20261115" to pin a snapshot. None = latest file.

# Surface grid points: output column -> (days, cp_flag, |delta| in OptionMetrics units).
# The stored SIGN of put deltas is detected from the data, not assumed.
SURFACE_GRID = {
    "atm_iv_30": (30, "C", 50),
    "iv_25p_30": (30, "P", 25),
    "iv_25c_30": (30, "C", 25),
    "atm_iv_91": (91, "C", 50),
}

TRADING_DAYS = 252                  # annualization factor
RV_WINDOWS = (5, 21, 63)            # past realized-variance windows (trading days)
TARGET_H = 21                       # forward window for the target (trading days)
RF_MAX_STALENESS_DAYS = 10          # calendar days a past zero-curve rate may be carried forward

HANDCHECK_DATES = ["2008-10-15", "2015-06-15", "2020-03-16"]   # snapped to the trading day on or before
N_LOOKAHEAD_CUTS = 6                # number of truncation dates in the look-ahead test
SHOW_PLOTS = False                  # True = open plot windows; PNGs are saved either way

TARGET_COLS = ["RV2_fwd21", "y"]    # the only columns allowed to use information after t

# Plot colors (blue / orange categorical pair, gray for reference lines)
C_BLUE, C_ORANGE, C_GRAY = "#2a78d6", "#eb6834", "#8a8a8a"


# =============================================================================
# SMALL HELPERS
# =============================================================================
def make_paths(data_dir: Path) -> dict[str, Path]:
    paths = {
        "raw": data_dir / "raw",
        "out": data_dir / "processed",
        "fig": data_dir / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def today_tag() -> str:
    return date.today().strftime("%Y%m%d")


def as_date(s: pd.Series) -> pd.Series:
    """Convert to datetime64[ns] at midnight so every merge key has the same dtype."""
    return pd.to_datetime(s).dt.normalize().astype("datetime64[ns]")


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save_frame(df: pd.DataFrame, path_no_ext: Path, fmt: str = FILE_FORMAT) -> Path:
    """Save as Parquet, or fall back to CSV if pyarrow is not installed."""
    if fmt == "parquet":
        try:
            path = path_no_ext.with_suffix(".parquet")
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            print("  pyarrow not installed: saving CSV instead of Parquet.")
    path = path_no_ext.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def load_raw(raw_dir: Path, name: str, tag: str | None = RAW_DATE_TAG) -> pd.DataFrame:
    """Load raw file '<name>_<YYYYMMDD>.(parquet|csv)'. Use the pinned tag, or else the latest."""
    files = [f for f in raw_dir.glob(f"{name}_*") if f.suffix in (".parquet", ".csv")]
    tagged = []
    for f in files:
        m = re.fullmatch(rf"{re.escape(name)}_(\d{{8}})", f.stem)
        if m:
            tagged.append((m.group(1), f))
    if tag is not None:
        tagged = [t for t in tagged if t[0] == tag]
    if not tagged:
        raise FileNotFoundError(
            f"No raw file for '{name}' (tag={tag}) in {raw_dir}. Run with --download first.")
    # Sort by tag; prefer parquet if both formats exist for the same tag.
    tagged.sort(key=lambda t: (t[0], t[1].suffix == ".parquet"))
    file_tag, path = tagged[-1]
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = as_date(df["date"])
    print(f"  loaded {path.name}: {len(df):,} rows, "
          f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    return df


# =============================================================================
# STEP 0 - DISCOVER WRDS LIBRARIES AND TABLES
# =============================================================================
def connect_wrds():
    """Open a WRDS connection. The package asks for username/password (and offers to
    create a ~/.pgpass file). No credentials are stored in this script."""
    import wrds  # imported here so --build works without the wrds package installed
    return wrds.Connection()


def available_libraries(db) -> list[str]:
    libs = set(db.list_libraries())
    found = [lib for lib in LIBRARY_CANDIDATES if lib in libs]
    if not found:
        raise RuntimeError(
            f"None of {LIBRARY_CANDIDATES} is visible to your account. "
            "Check your OptionMetrics subscription on WRDS.")
    return found


def print_tables(db, libraries: list[str]) -> None:
    """Print every table in each library. Year-split families are shown as a year range."""
    header("STEP 0 - OptionMetrics tables visible on WRDS")
    for lib in libraries:
        tables = sorted(db.list_tables(library=lib))
        families, singles = defaultdict(list), []
        for t in tables:
            m = re.fullmatch(r"([a-z_]+?)(\d{4})", t)
            if m:
                families[m.group(1)].append(int(m.group(2)))
            else:
                singles.append(t)
        print(f"\n[{lib}] {len(tables)} tables")
        for fam, years in sorted(families.items()):
            print(f"  {fam}YYYY : {min(years)}-{max(years)} ({len(years)} tables)")
        print("  single tables:")
        print(textwrap.indent(textwrap.fill(", ".join(singles), 74), "    "))
        print("  full raw list:")
        print(textwrap.indent(textwrap.fill(", ".join(tables), 74), "    "))


def find_tables(db, libraries: list[str], prefix: str) -> tuple[str, list[str]]:
    """Return (library, tables) for a family. Year-split tables are preferred and sorted
    by year. Otherwise use a single stacked table named exactly `prefix`."""
    for lib in libraries:
        tables = db.list_tables(library=lib)
        split = sorted(
            (int(m.group(1)), t) for t in tables
            if (m := re.fullmatch(rf"{prefix}(\d{{4}})", t)))
        if split:
            return lib, [t for _, t in split]
        if prefix in tables:
            return lib, [prefix]
    raise RuntimeError(f"No table '{prefix}' or '{prefix}YYYY' found in {libraries}.")


def resolve_columns(db, lib: str, tables: list[str], required: list[str],
                    optional: list[str] = ()) -> list[str]:
    """Check column names with describe_table on the first and last table of a family,
    because a column can be added in later years. Stop if a required column is missing."""
    present_optional = set(optional)
    for t in {tables[0], tables[-1]}:
        cols = set(db.describe_table(library=lib, table=t)["name"].str.lower())
        missing = [c for c in required if c not in cols]
        if missing:
            raise RuntimeError(f"{lib}.{t} is missing required columns {missing}. "
                               f"Available columns: {sorted(cols)}")
        present_optional &= cols
        print(f"  {lib}.{t} columns: {sorted(cols)}")
    chosen = list(required) + [c for c in optional if c in present_optional]
    skipped = [c for c in optional if c not in present_optional]
    if skipped:
        print(f"  optional columns not available in every table, skipped: {skipped}")
    return chosen


def date_filter_sql() -> str:
    sql = ""
    if START_DATE:
        sql += f" AND date >= '{pd.Timestamp(START_DATE):%Y-%m-%d}'"
    if END_DATE:
        sql += f" AND date <= '{pd.Timestamp(END_DATE):%Y-%m-%d}'"
    return sql


def pull_stacked(db, lib: str, tables: list[str], cols: list[str], where: str) -> pd.DataFrame:
    """Query each (yearly) table with the same SELECT/WHERE and stack the results.
    Table names come from list_tables and secid is cast to int, so the string
    formatting cannot inject SQL."""
    frames = []
    for t in tables:
        sql = f"SELECT {', '.join(cols)} FROM {lib}.{t} WHERE {where}"
        df = db.raw_sql(sql, date_cols=["date"])
        print(f"  {lib}.{t:<12} {len(df):>9,} rows")
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No rows returned from {lib}.{tables[0]}... with WHERE {where}")
    out = pd.concat(frames, ignore_index=True).sort_values("date", ignore_index=True)
    return out


# =============================================================================
# STEP 1 - DOWNLOAD RAW EXTRACTS (WRDS + FRED) TO DISK
# =============================================================================
def download_vix() -> pd.DataFrame:
    """Download VIXCLS from FRED as CSV. Fallback: pandas_datareader."""
    try:
        v = pd.read_csv(FRED_VIX_URL, na_values=["."])
        v = v.rename(columns={v.columns[0]: "date"})       # 'observation_date' or 'DATE'
    except Exception as e:                                   # network / SSL problems
        print(f"  FRED CSV download failed ({e}); trying pandas_datareader...")
        import pandas_datareader.data as web
        v = web.DataReader("VIXCLS", "fred", start="1990-01-01").reset_index()
        v = v.rename(columns={v.columns[0]: "date"})
    if "VIXCLS" not in v.columns:
        raise RuntimeError(f"Unexpected FRED columns: {list(v.columns)}")
    v["date"] = as_date(v["date"])
    v["VIXCLS"] = pd.to_numeric(v["VIXCLS"], errors="coerce")
    return v[["date", "VIXCLS"]]


def run_download(db, libraries: list[str], paths: dict[str, Path]) -> None:
    header("STEP 1 - Download raw extracts")
    tag = today_tag()
    manifest = {"download_tag": tag, "secid": SECID, "sources": {}}
    secid_where = f"secid = {int(SECID)}" + date_filter_sql()

    jobs = [
        ("surface", SURFACE_PREFIX, SURFACE_COLS_REQUIRED, SURFACE_COLS_OPTIONAL, secid_where),
        ("prices", PRICE_PREFIX, PRICE_COLS_REQUIRED, [], secid_where),
        # The zero curve is not tied to a security, so there is no secid filter.
        ("zero", ZERO_PREFIX, ZERO_COLS_REQUIRED, [], "1=1" + date_filter_sql()),
    ]
    for key, prefix, req, opt, where in jobs:
        lib, tables = find_tables(db, libraries, prefix)
        print(f"\n[{key}] {lib}.{prefix}: {len(tables)} table(s) {tables[0]} ... {tables[-1]}")
        cols = resolve_columns(db, lib, tables, req, opt)
        df = pull_stacked(db, lib, tables, cols, where)
        name = raw_name(key)
        path = save_frame(df, paths["raw"] / f"{name}_{tag}")
        print(f"  -> {path.name}: {len(df):,} rows, first date {df['date'].min():%Y-%m-%d}, "
              f"last date {df['date'].max():%Y-%m-%d}")
        manifest["sources"][key] = {
            "library": lib, "tables": tables, "columns": cols, "where": where,
            "rows": len(df), "first_date": f"{df['date'].min():%Y-%m-%d}",
            "last_date": f"{df['date'].max():%Y-%m-%d}", "file": path.name}

    print("\n[vix] FRED VIXCLS")
    vix = download_vix()
    path = save_frame(vix, paths["raw"] / f"{raw_name('vix')}_{tag}")
    print(f"  -> {path.name}: {len(vix):,} rows, {vix['date'].min():%Y-%m-%d} -> "
          f"{vix['date'].max():%Y-%m-%d}")
    manifest["sources"]["vix"] = {"url": FRED_VIX_URL, "rows": len(vix), "file": path.name}

    with open(paths["raw"] / f"manifest_{tag}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: manifest_{tag}.json")


def raw_name(key: str) -> str:
    return {
        "surface": f"optionm_vsurfd_secid{SECID}",
        "prices": f"optionm_secprd_secid{SECID}",
        "zero": "optionm_zerocd",
        "vix": "fred_vixcls",
    }[key]


# =============================================================================
# STEP 2a - INSPECT RAW DATA (units, conventions)
# =============================================================================
def clean_surface(surf: pd.DataFrame) -> pd.DataFrame:
    s = surf.copy()
    s["cp_flag"] = s["cp_flag"].astype(str).str.strip().str.upper()
    s["impl_volatility"] = pd.to_numeric(s["impl_volatility"], errors="coerce")
    # OptionMetrics can use negative sentinels (e.g. -99.99) for missing values.
    s.loc[s["impl_volatility"] <= 0, "impl_volatility"] = np.nan
    return s


def detect_put_delta_sign(surf: pd.DataFrame) -> int:
    """Return -1 if put deltas are stored as negative numbers (-25), +1 if they are
    stored as absolute values (25). Stop on mixed or unexpected conventions."""
    s = clean_surface(surf)
    put_d = s.loc[s["cp_flag"] == "P", "delta"].dropna()
    call_d = s.loc[s["cp_flag"] == "C", "delta"].dropna()
    if not (call_d > 0).all():
        raise ValueError("Some call deltas are <= 0. Check the delta convention.")
    if (put_d < 0).all():
        return -1
    if (put_d > 0).all():
        return +1
    raise ValueError("Put deltas have mixed signs. Check the delta convention by year.")


def inspect_surface(surf: pd.DataFrame) -> None:
    header("STEP 2a - Surface conventions (verify these!)")
    s = clean_surface(surf)
    assert (s["secid"] == SECID).all(), "surface contains other secids"
    print("cp_flag values      :", sorted(s["cp_flag"].unique()))
    print("distinct days       :", sorted(s["days"].dropna().unique().tolist()))
    for cp in ("C", "P"):
        vals = sorted(s.loc[s["cp_flag"] == cp, "delta"].dropna().unique().tolist())
        print(f"distinct deltas ({cp})  :", vals)
    # Check that the put-delta sign convention is the same every year.
    signs = (s[s["cp_flag"] == "P"].groupby(s["date"].dt.year)["delta"]
             .agg(lambda d: "neg" if (d < 0).all() else ("pos" if (d > 0).all() else "mixed")))
    print("put-delta sign, number of years:", {k: int(v) for k, v in signs.value_counts().items()})
    sign = detect_put_delta_sign(surf)
    print(f"=> put deltas are stored as {'NEGATIVE' if sign < 0 else 'POSITIVE'} numbers, so "
          f"the 25-delta put is delta = {25 * sign}")
    iv = s["impl_volatility"]
    print(f"impl_volatility: median {iv.median():.4f}, min {iv.min():.4f}, max {iv.max():.4f}, "
          f"NaN/invalid {iv.isna().sum():,}  -> expect decimals (0.20 = 20%)")
    if iv.median() > 2:
        raise ValueError("Median IV > 2: the IVs look like percent, not decimals. Check units.")
    print(f"surface dates: first {s['date'].min():%Y-%m-%d}, last {s['date'].max():%Y-%m-%d}, "
          f"{s['date'].nunique():,} distinct days")


# =============================================================================
# STEP 2b - BUILD FEATURES. Each block uses data dated <= t only.
# =============================================================================
def surface_features(surf: pd.DataFrame, put_sign: int) -> pd.DataFrame:
    """Pick the four grid points and make one row per date. Each value comes from
    that date's surface only (row-wise filter + pivot, no time operations)."""
    s = clean_surface(surf)
    cols = []
    for col, (days, cp, absdelta) in SURFACE_GRID.items():
        stored_delta = absdelta * (put_sign if cp == "P" else 1)
        m = (s["days"] == days) & (s["cp_flag"] == cp) & np.isclose(s["delta"], stored_delta)
        sub = s.loc[m, ["date", "impl_volatility"]]
        if sub.empty:
            raise ValueError(f"Grid point {col} (days={days}, {cp}, delta={stored_delta}) "
                             "not found. Check the printed days/deltas.")
        dup = sub["date"].duplicated(keep=False)
        if dup.any():
            conflicting = sub[dup].groupby("date")["impl_volatility"].nunique(dropna=False) > 1
            if conflicting.any():
                raise ValueError(f"{col}: conflicting duplicate rows on "
                                 f"{conflicting[conflicting].index[:5].tolist()}")
            sub = sub.drop_duplicates("date")
        cols.append(sub.set_index("date")["impl_volatility"].rename(col))
    # An outer join on date leaves NaN when a grid point is missing that day. Nothing is filled.
    return pd.concat(cols, axis=1, sort=True)


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, past realized variances, and the forward target, on the
    SPX trading calendar (the rows of secprd)."""
    p = prices[["date", "close"]].copy()
    if p["date"].duplicated().any():
        raise ValueError("Duplicate dates in secprd for this secid.")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    n_neg = int((p["close"] < 0).sum())
    if n_neg:
        # Some providers store a bid/ask midpoint as a negative close. Verify for SPX.
        print(f"  WARNING: {n_neg} negative closes; using absolute values.")
        p["close"] = p["close"].abs()
    p.loc[p["close"] == 0, "close"] = np.nan
    p = p.sort_values("date", ignore_index=True)

    # r_t = ln(S_t / S_{t-1}): uses the close of t and t-1 only.
    # A missing close gives NaN returns, never a hidden multi-day return.
    p["r"] = np.log(p["close"]).diff()
    r2 = p["r"] ** 2

    # Past realized variance: rolling window ENDING at t (rows t-h+1..t), right-aligned.
    for h in RV_WINDOWS:
        p[f"RV2_{h}"] = (TRADING_DAYS / h) * r2.rolling(h, min_periods=h).sum()

    # Target: sum of r^2 over t+1..t+H. rolling(H) at row t+H covers t+1..t+H,
    # and shift(-H) moves that value back to row t. The last H rows become NaN.
    fwd_sum = r2.rolling(TARGET_H, min_periods=TARGET_H).sum().shift(-TARGET_H)
    p["RV2_fwd21"] = (TRADING_DAYS / TARGET_H) * fwd_sum
    p["y"] = np.log(p["RV2_fwd21"].where(p["RV2_fwd21"] > 0))
    return p


def risk_free(zero: pd.DataFrame) -> pd.DataFrame:
    """Shortest-maturity zero rate per date, converted from percent to decimal."""
    z = zero[["date", "days", "rate"]].dropna()
    z = z.loc[z.groupby("date")["days"].idxmin()]
    return pd.DataFrame({
        "date": z["date"].to_numpy(),
        "rf": z["rate"].to_numpy() / 100.0,          # percent -> decimal
        "rf_days": z["days"].to_numpy(),
        "rf_source_date": z["date"].to_numpy(),
    }).sort_values("date", ignore_index=True)


def build_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Turn the raw frames into the daily panel. This is a pure function of its
    inputs, which is what makes the truncation (look-ahead) test possible."""
    put_sign = detect_put_delta_sign(raw["surface"])
    surf = surface_features(raw["surface"], put_sign)

    prices = raw["prices"]
    if END_DATE:
        prices = prices[prices["date"] <= pd.Timestamp(END_DATE)]
    px = price_features(prices)

    # Trading calendar = SPX price dates, starting on the first day with a surface.
    # (Returns/RV were computed before this trim, so day 1 can use the prior close.)
    start = max(surf.index.min(), pd.Timestamp(START_DATE)) if START_DATE else surf.index.min()
    panel = px[px["date"] >= start].reset_index(drop=True)

    # Surface: exact-date left join. A missing grid point stays NaN.
    panel = panel.merge(surf, left_on="date", right_index=True, how="left")
    panel["skew"] = panel["iv_25p_30"] - panel["iv_25c_30"]
    panel["curvature"] = panel["iv_25p_30"] + panel["iv_25c_30"] - 2 * panel["atm_iv_30"]
    panel["term_slope"] = panel["atm_iv_91"] - panel["atm_iv_30"]

    # Risk-free: backward as-of merge = latest rate dated <= t (a forward fill with
    # past values only), limited to RF_MAX_STALENESS_DAYS.
    rf = risk_free(raw["zero"])
    panel = pd.merge_asof(panel.sort_values("date"), rf, on="date", direction="backward",
                          tolerance=pd.Timedelta(days=RF_MAX_STALENESS_DAYS))
    panel["rf_filled"] = panel["rf_source_date"].notna() & (panel["rf_source_date"] < panel["date"])
    panel = panel.drop(columns="rf_source_date")

    # VIX: exact-date left join, in points and as a decimal. Not filled.
    vix = raw["vix"][["date", "VIXCLS"]].rename(columns={"VIXCLS": "vix_pts"})
    panel = panel.merge(vix, on="date", how="left")
    panel["vix"] = panel["vix_pts"] / 100.0

    order = (["date", "close", "r"] + list(SURFACE_GRID) + ["skew", "curvature", "term_slope"]
             + [f"RV2_{h}" for h in RV_WINDOWS] + ["rf", "rf_days", "rf_filled", "vix_pts", "vix"]
             + TARGET_COLS)
    return panel[order].reset_index(drop=True)


def feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in TARGET_COLS + ["date"]]


# =============================================================================
# STEP 3 - CHECKS
# =============================================================================
def check_coverage(panel: pd.DataFrame, paths) -> None:
    header("CHECK 1 - Rows and missing values per year")
    yr = panel["date"].dt.year
    print("rows per year:\n" + panel.groupby(yr).size().to_string())
    miss = panel.drop(columns="date").isna().groupby(yr).sum()
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\nmissing values per column per year:\n", miss)
    miss.to_csv(paths["out"] / "missing_by_year.csv")
    gaps = panel["date"].diff().dt.days
    print("\nlargest calendar gaps between consecutive trading days (expect 9/11 2001, Sandy 2012):")
    print(panel.loc[gaps.nlargest(5).index, "date"].dt.strftime("%Y-%m-%d").to_frame()
          .assign(gap_days=gaps.nlargest(5).values).to_string(index=False))


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def check_vix(panel: pd.DataFrame, paths) -> None:
    import matplotlib.pyplot as plt
    header("CHECK 2 - atm_iv_30 vs VIX/100")
    d = panel[["date", "atm_iv_30", "vix"]].dropna()
    corr_lvl = d["atm_iv_30"].corr(d["vix"])
    corr_chg = d["atm_iv_30"].diff().corr(d["vix"].diff())
    print(f"overlapping days: {len(d):,}")
    print(f"correlation (levels)       : {corr_lvl:.4f}   (expect >= 0.95)")
    print(f"correlation (daily changes): {corr_chg:.4f}   (stricter test; expect high)")
    print(f"mean(VIX/100 - atm_iv_30)  : {(d['vix'] - d['atm_iv_30']).mean():.4f} "
          "(VIX > ATM IV is normal: VIX also prices OTM puts)")
    print("PASS" if corr_lvl >= 0.95 else "WARNING: correlation below 0.95. Check units/grid.")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["date"], d["vix"], color=C_ORANGE, lw=1.0, label="VIX / 100 (FRED)")
    ax.plot(d["date"], d["atm_iv_30"], color=C_BLUE, lw=1.0, label="atm_iv_30 (OptionMetrics)")
    _style(ax, f"30-day ATM implied vol vs VIX (corr = {corr_lvl:.3f})", "annualized vol (decimal)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths["fig"] / "atm_iv_30_vs_vix.png", dpi=150)


def check_skew(panel: pd.DataFrame, paths) -> None:
    import matplotlib.pyplot as plt
    header("CHECK 3 - Skew sign (25d put IV - 25d call IV)")
    s = panel["skew"].dropna()
    share = (s > 0).mean()
    print(f"share of days with skew > 0: {share:.2%}  (expect most days)")
    print("by year:\n" + (panel.dropna(subset=["skew"]).groupby(panel["date"].dt.year)["skew"]
                         .apply(lambda x: (x > 0).mean()).map("{:.1%}".format).to_string()))
    print("PASS" if share > 0.9 else "WARNING: skew not positive on most days. Check put/call deltas.")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(panel["date"], panel["skew"], color=C_BLUE, lw=0.8)
    ax.axhline(0, color=C_GRAY, lw=0.8)
    _style(ax, "30-day skew: iv_25p_30 - iv_25c_30", "vol points (decimal)")
    fig.tight_layout()
    fig.savefig(paths["fig"] / "skew.png", dpi=150)


def check_term_slope(panel: pd.DataFrame, paths) -> None:
    import matplotlib.pyplot as plt
    header("CHECK 4 - Term slope (atm_iv_91 - atm_iv_30) in crises")
    episodes = {"GFC Sep-Dec 2008": ("2008-09-15", "2008-12-31"),
                "COVID Mar 2020": ("2020-03-01", "2020-03-31")}
    ts = panel.set_index("date")["term_slope"]
    print(f"full-sample share negative: {(ts.dropna() < 0).mean():.1%}")
    for name, (a, b) in episodes.items():
        w = ts.loc[a:b].dropna()
        if w.empty:
            print(f"{name}: no data")
            continue
        print(f"{name}: mean {w.mean():+.4f}, share negative {(w < 0).mean():.0%}, "
              f"min {w.min():+.4f} on {w.idxmin():%Y-%m-%d} -> "
              f"{'PASS (inverted)' if w.mean() < 0 else 'WARNING: not inverted'}")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for a, b in episodes.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d9d9", alpha=0.6, lw=0)
    ax.plot(panel["date"], panel["term_slope"], color=C_BLUE, lw=0.8)
    ax.axhline(0, color=C_GRAY, lw=0.8)
    _style(ax, "Term slope: atm_iv_91 - atm_iv_30 (shaded: Sep-Dec 2008, Mar 2020)",
           "vol points (decimal)")
    fig.tight_layout()
    fig.savefig(paths["fig"] / "term_slope.png", dpi=150)


def hand_check(panel: pd.DataFrame, paths) -> None:
    """Print RV and target for chosen dates, and export the window rows to CSV for Excel."""
    header("CHECK 5 - Hand-check dates (verify in Excel)")
    cal = panel["date"]
    for d in HANDCHECK_DATES:
        eligible = cal[cal <= pd.Timestamp(d)]
        if eligible.empty:
            print(f"{d}: before sample start, skipped")
            continue
        t = eligible.idxmax()                 # row of the trading day on or before d
        lo, hi = max(0, t - max(RV_WINDOWS)), min(len(panel) - 1, t + TARGET_H)
        win = panel.loc[lo:hi, ["date", "close", "r"]].copy()
        win.insert(0, "offset", win.index - t)
        win["r2"] = win["r"] ** 2
        row = panel.loc[t]
        print(f"\ndate t = {row['date']:%Y-%m-%d} (requested {d})")
        for h in RV_WINDOWS:
            manual = TRADING_DAYS / h * win.loc[win["offset"].between(-h + 1, 0), "r2"].sum()
            print(f"  RV2_{h:<3}= {row[f'RV2_{h}']:.6f}   manual check {manual:.6f}")
        manual_fwd = TRADING_DAYS / TARGET_H * win.loc[win["offset"].between(1, TARGET_H), "r2"].sum()
        print(f"  RV2_fwd21 = {row['RV2_fwd21']:.6f}   manual check {manual_fwd:.6f}")
        print(f"  y         = {row['y']:.6f}")
        out = paths["out"] / f"handcheck_{row['date']:%Y%m%d}.csv"
        win.to_csv(out, index=False)
        print(f"  window rows -> {out.name}")
    print(textwrap.dedent(f"""
      Excel recipe (each CSV: offset 0 = day t, columns offset/date/close/r/r2):
        r         = LN(close_t / close_(t-1))            -> compare with column r
        RV2_21    = 252/21 * SUMSQ(r for offset -20..0)
        RV2_5     = 252/5  * SUMSQ(r for offset  -4..0)
        RV2_63    = 252/63 * SUMSQ(r for offset -62..0)
        RV2_fwd21 = 252/21 * SUMSQ(r for offset  +1..+21)
        y         = LN(RV2_fwd21)"""))


def assert_no_lookahead(raw: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    """Two tests:
    (A) Truncation: rebuild the panel from raw data cut at date c. Every feature
        dated <= c must be identical to the full-sample build. If any feature used
        data after t, removing the future would change it and the test would fail.
    (B) Window alignment: recompute RV and the target for random rows with an
        explicit numpy loop and check that the windows are exactly t-h+1..t and
        t+1..t+21. Also check that the last 21 targets are NaN.
    """
    header("CHECK 6 - No look-ahead assertions")
    feats = feature_cols(panel)
    cal = panel["date"]

    # (A) truncation test
    qs = np.linspace(0.15, 0.95, N_LOOKAHEAD_CUTS)
    cuts = set(cal.iloc[(qs * (len(cal) - 1)).astype(int)])
    for d in HANDCHECK_DATES:
        e = cal[cal <= pd.Timestamp(d)]
        if not e.empty:
            cuts.add(e.max())
    for c in sorted(cuts):
        raw_c = {k: v[v["date"] <= c] for k, v in raw.items()}
        p_c = build_panel(raw_c).set_index("date")[feats]
        p_full = panel[panel["date"] <= c].set_index("date")[feats]
        assert p_c.index.equals(p_full.index), f"calendar differs at cut {c:%Y-%m-%d}"
        pd.testing.assert_frame_equal(p_c, p_full, check_exact=False, rtol=1e-10,
                                      obj=f"features with data cut at {c:%Y-%m-%d}")
        assert p_c.index[-1] == c
        print(f"  (A) cut {c:%Y-%m-%d}: all {len(feats)} feature columns identical  OK")

    # Power check: the test must FAIL for a deliberately leaky feature. Tomorrow's
    # close (close.shift(-1)) differs between the full and truncated builds at the cut.
    c = sorted(cuts)[0]
    leak_full = panel.set_index("date")["close"].shift(-1).loc[:c]
    leak_cut = build_panel({k: v[v["date"] <= c] for k, v in raw.items()}) \
        .set_index("date")["close"].shift(-1)
    assert not leak_full.equals(leak_cut), "truncation test cannot detect a leak"
    print("  (A) power check: a leaky feature (close at t+1) is detected  OK")

    # (B) window alignment against an explicit loop
    r = panel["r"].to_numpy()
    n = len(r)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(max(RV_WINDOWS), n), size=min(300, n - max(RV_WINDOWS)), replace=False)
    for i in idx:
        for h in RV_WINDOWS:
            exp = TRADING_DAYS / h * np.sum(r[i - h + 1:i + 1] ** 2)  # NaN if any r is NaN
            got = panel.at[i, f"RV2_{h}"]
            assert np.isclose(got, exp, rtol=1e-9, equal_nan=True), f"RV2_{h} misaligned at row {i}"
        exp_t = (TRADING_DAYS / TARGET_H * np.sum(r[i + 1:i + TARGET_H + 1] ** 2)
                 if i + TARGET_H < n else np.nan)
        assert np.isclose(panel.at[i, "RV2_fwd21"], exp_t, rtol=1e-9, equal_nan=True), \
            f"target misaligned at row {i}"
    print(f"  (B) {len(idx)} random rows: RV windows = t-h+1..t, target window = t+1..t+21  OK")

    assert panel["RV2_fwd21"].iloc[-TARGET_H:].isna().all() and panel["y"].iloc[-TARGET_H:].isna().all()
    assert not set(TARGET_COLS) & set(feats)
    print(f"  last {TARGET_H} rows have NaN targets; target columns are not in the feature list  OK")
    print("ALL LOOK-AHEAD ASSERTIONS PASSED")


# =============================================================================
# STEP 2 DRIVER
# =============================================================================
def run_build(paths: dict[str, Path]) -> pd.DataFrame:
    header("STEP 2 - Build panel from raw files (no WRDS access)")
    raw = {k: load_raw(paths["raw"], raw_name(k)) for k in ("surface", "prices", "zero", "vix")}
    assert (raw["prices"]["secid"] == SECID).all(), "prices contain other secids"

    inspect_surface(raw["surface"])
    rf = risk_free(raw["zero"])
    print("\nzero curve: shortest maturity used (days -> number of dates):",
          {int(k): int(v) for k, v in rf["rf_days"].value_counts().sort_index().head(10).items()})
    print(f"rf (decimal): median {rf['rf'].median():.4f}, min {rf['rf'].min():.4f}, "
          f"max {rf['rf'].max():.4f}  -> expect about 0.00-0.07")

    panel = build_panel(raw)
    print(f"\nPANEL: {len(panel):,} rows, first date {panel['date'].min():%Y-%m-%d}, "
          f"last date {panel['date'].max():%Y-%m-%d}")
    print(f"  surface range: {raw['surface']['date'].min():%Y-%m-%d} -> "
          f"{raw['surface']['date'].max():%Y-%m-%d}; prices range: "
          f"{raw['prices']['date'].min():%Y-%m-%d} -> {raw['prices']['date'].max():%Y-%m-%d}")
    surf_dates = set(raw["surface"]["date"].unique())
    print(f"  surface dates not on the price calendar (dropped): "
          f"{len(surf_dates - set(panel['date'])):,}")
    print(f"  rf filled from an earlier date on {int(panel['rf_filled'].sum()):,} days")

    tag = today_tag()
    p1 = save_frame(panel, paths["out"] / f"spx_daily_panel_{tag}")
    panel.to_csv(paths["out"] / f"spx_daily_panel_{tag}.csv", index=False)
    print(f"  saved {p1.name} (+ CSV copy)")

    check_coverage(panel, paths)
    check_vix(panel, paths)
    check_skew(panel, paths)
    check_term_slope(panel, paths)
    hand_check(panel, paths)
    assert_no_lookahead(raw, panel)
    print(f"\nFigures saved in {paths['fig']}")
    return panel


def export_raw_csv(paths: dict[str, Path]) -> None:
    """Write a CSV copy of each raw table (one file per source), next to the Parquet files.
    Note: the surface CSV exceeds Excel's 1,048,576-row limit; open it in Python instead."""
    header("Export raw tables to CSV")
    for key in ("surface", "prices", "zero", "vix"):
        name = raw_name(key)
        df = load_raw(paths["raw"], name)
        tag = sorted(re.findall(rf"{re.escape(name)}_(\d{{8}})", " ".join(
            f.stem for f in paths["raw"].glob(f"{name}_*.parquet"))))[-1]
        out = paths["raw"] / f"{name}_{tag}.csv"
        df.to_csv(out, index=False, date_format="%Y-%m-%d")
        print(f"  -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-tables", action="store_true", help="print WRDS tables and stop")
    ap.add_argument("--download", action="store_true", help="download raw extracts")
    ap.add_argument("--build", action="store_true", help="build panel from raw files")
    ap.add_argument("--export-raw-csv", action="store_true",
                    help="write a CSV copy of each raw table from the saved Parquet files, then stop")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = make_paths(args.data_dir)
    if args.export_raw_csv:
        export_raw_csv(paths)
        return
    run_all = not (args.list_tables or args.download or args.build)

    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")

    if run_all or args.list_tables or args.download:
        db = connect_wrds()
        try:
            libs = available_libraries(db)
            print_tables(db, libs)
            if run_all or args.download:
                run_download(db, libs, paths)
        finally:
            db.close()

    if run_all or args.build:
        run_build(paths)
        if SHOW_PLOTS:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()

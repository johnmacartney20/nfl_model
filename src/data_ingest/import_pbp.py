# pulls play-by-play for selected seasons
import ssl
import nfl_data_py as nfl
import pandas as pd
from src.utils.config import RAW_DIR
from src.utils.helpers import ensure_dirs

def import_pbp_data(seasons):
    ensure_dirs([RAW_DIR])
    
    # Fix SSL certificate verification issue on macOS
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print(f"Loading pbp for seasons: {seasons}")
    pbp = nfl.import_pbp_data(seasons)

    # Keep only the columns we actually need downstream to cut memory/disk usage.
    keep_cols = [
        "season",
        "season_type",
        "week",
        "game_id",
        "play_id",
        "play_type",
        "posteam",
        "defteam",
        "epa",
        "success",
    ]
    available = [c for c in keep_cols if c in pbp.columns]
    if available:
        pbp = pbp[available].copy()
    # Filter to the play types we use (feature engineering re-checks this as well).
    if "play_type" in pbp.columns:
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    out_path = RAW_DIR / f"pbp_{min(seasons)}_{max(seasons)}.parquet"
    pbp.to_parquet(out_path, index=False)
    print(f"Saved pbp to {out_path}")
    return pbp

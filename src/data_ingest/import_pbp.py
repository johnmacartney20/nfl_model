# pulls play-by-play for selected seasons
import ssl

import nfl_data_py as nfl
import pandas as pd
import requests

from src.utils.config import RAW_DIR
from src.utils.helpers import ensure_dirs


if not hasattr(nfl, "Error"):
    nfl.Error = Exception


def _resolve_available_pbp_seasons(seasons):
    """Return the subset of requested seasons whose nflverse parquet files exist."""
    available = []
    for year in sorted(set(int(season) for season in seasons)):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
        try:
            response = requests.head(url, allow_redirects=True, timeout=20)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            available.append(year)
    return available


def import_pbp_data(seasons):
    ensure_dirs([RAW_DIR])

    seasons = sorted(set(int(season) for season in seasons))
    available_seasons = _resolve_available_pbp_seasons(seasons)
    missing_seasons = [season for season in seasons if season not in available_seasons]
    if missing_seasons:
        print(f"Skipping unavailable play-by-play data for seasons: {missing_seasons}")
    if not available_seasons:
        raise ValueError("No play-by-play data is available for the requested seasons.")

    # Fix SSL certificate verification issue on macOS
    ssl._create_default_https_context = ssl._create_unverified_context

    print(f"Loading pbp for seasons: {available_seasons}")
    pbp = nfl.import_pbp_data(available_seasons, include_participation=False)

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

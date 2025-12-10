# loads DVOA data from a pre-saved CSV
import pandas as pd
from src.utils.config import RAW_DIR
from src.utils.helpers import ensure_dirs


def import_dvoa_data(seasons):
    """
    Load DVOA data for the given seasons.

    Expects a CSV like RAW_DIR / "dvoa_all.csv" with at least:
      season, week, team, off_dvoa, def_dvoa, st_dvoa

    You can expand this later if you have more detailed DVOA splits.
    """
    ensure_dirs([RAW_DIR])

    dvoa_path = RAW_DIR / "dvoa_all.csv"
    dvoa = pd.read_csv(dvoa_path)

    # basic sanity filter
    dvoa = dvoa[dvoa["season"].isin(seasons)].copy()

    # normalize team column name just in case
    dvoa = dvoa.rename(columns={"team": "team"})

    print(f"Loaded DVOA from {dvoa_path} for seasons {seasons}")
    return dvoa

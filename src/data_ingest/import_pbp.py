# pulls play-by-play for selected seasons
import nfl_data_py as nfl
import pandas as pd
from src.utils.config import RAW_DIR
from src.utils.helpers import ensure_dirs

def import_pbp_data(seasons):
    ensure_dirs([RAW_DIR])
    print(f"Loading pbp for seasons: {seasons}")
    pbp = nfl.import_pbp_data(seasons)
    out_path = RAW_DIR / f"pbp_{min(seasons)}_{max(seasons)}.parquet"
    pbp.to_parquet(out_path, index=False)
    print(f"Saved pbp to {out_path}")
    return pbp

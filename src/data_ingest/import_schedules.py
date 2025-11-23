# pulls games/schedules
import nfl_data_py as nfl
import pandas as pd
from src.utils.config import RAW_DIR
from src.utils.helpers import ensure_dirs

def import_schedules(seasons):
    ensure_dirs([RAW_DIR])
    print(f"Loading schedules for seasons: {seasons}")
    sched = nfl.import_schedules(seasons)
    # Keep only regular and playoff games for now
    sched = sched[sched["game_type"].isin(["REG", "POST"])]
    # Rename a few columns to keep things clear
    sched = sched.rename(
        columns={
            "home_team": "home_team",
            "away_team": "away_team",
            "home_score": "home_score",
            "away_score": "away_score",
        }
    )
    out_path = RAW_DIR / f"schedules_{min(seasons)}_{max(seasons)}.csv"
    sched.to_csv(out_path, index=False)
    print(f"Saved schedules to {out_path}")
    return sched

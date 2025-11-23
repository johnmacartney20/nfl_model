import pandas as pd

from src.utils.config import (
    RAW_DIR,
    TEAM_GAME_EPA_CSV,
    FEATURE_DIR,
    GAME_LEVEL_FEATURES_CSV,
)
from src.utils.helpers import ensure_dirs, add_home_away_flags

def build_game_level_features(seasons):
    ensure_dirs([FEATURE_DIR])

    # Load schedule
    sched_path = RAW_DIR / f"schedules_{min(seasons)}_{max(seasons)}.csv"
    sched = pd.read_csv(sched_path)

    # Load season-to-date EPA
    STD_EPA_CSV = RAW_DIR.parent / "interim" / "season_to_date_epa.csv"
    season_epa = pd.read_csv(STD_EPA_CSV)

    # Merge home team’s season-to-date EPA
    df = sched.merge(
        season_epa.rename(
            columns={
                "team": "home_team",
                "team_avg_off_epa": "home_avg_off_epa",
                "team_avg_def_epa": "home_avg_def_epa",
                "team_avg_success_rate": "home_avg_success_rate",
            }
        ),
        on=["season", "home_team"],
        how="left",
    )

    # Merge away team’s season-to-date EPA
    df = df.merge(
        season_epa.rename(
            columns={
                "team": "away_team",
                "team_avg_off_epa": "away_avg_off_epa",
                "team_avg_def_epa": "away_avg_def_epa",
                "team_avg_success_rate": "away_avg_success_rate",
            }
        ),
        on=["season", "away_team"],
        how="left",
    )

    df = add_home_away_flags(df)

    df.to_csv(GAME_LEVEL_FEATURES_CSV, index=False)
    print(f"Saved game level features to {GAME_LEVEL_FEATURES_CSV}")

    return df

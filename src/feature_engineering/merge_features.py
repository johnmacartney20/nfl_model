import pandas as pd

from src.utils.config import (
    RAW_DIR,
    TEAM_GAME_EPA_CSV,
    FEATURE_DIR,
    GAME_LEVEL_FEATURES_CSV,
    get_excluded_reg_season_weeks_by_season,
)
from src.utils.helpers import ensure_dirs, add_home_away_flags
from src.feature_engineering.build_team_game_epa import compute_pregame_team_features


def build_game_level_features(seasons):
    ensure_dirs([FEATURE_DIR])

    # Load schedule
    sched_path = RAW_DIR / f"schedules_{min(seasons)}_{max(seasons)}.csv"
    sched = pd.read_csv(sched_path)

    # Optionally exclude noisy late-season regular-season weeks (e.g., rested starters)
    excluded_weeks = get_excluded_reg_season_weeks_by_season()
    if excluded_weeks and {"season", "week", "game_type"}.issubset(set(sched.columns)):
        exclude_mask = pd.Series(False, index=sched.index)
        for season, weeks in excluded_weeks.items():
            if not weeks:
                continue
            exclude_mask |= (
                (sched["season"] == season)
                & (sched["game_type"] == "REG")
                & (sched["week"].isin(weeks))
            )
        sched = sched[~exclude_mask].copy()

    # Point-in-time team ratings: each game sees only games played before it.
    # (Previously full-season averages were joined on (season, team), which
    # leaked each game's own result and every later game into its features.)
    team_game_epa = pd.read_csv(TEAM_GAME_EPA_CSV)
    pregame = compute_pregame_team_features(sched, team_game_epa)

    def _side(prefix):
        return pregame.drop(columns=["season", "week"]).rename(
            columns={
                "team": f"{prefix}_team",
                "team_avg_off_epa": f"{prefix}_avg_off_epa",
                "team_avg_def_epa": f"{prefix}_avg_def_epa",
                "team_avg_success_rate": f"{prefix}_avg_success_rate",
                "team_games_played_pre": f"{prefix}_games_played_pre",
                "off_dvoa": f"{prefix}_off_dvoa",
                "def_dvoa": f"{prefix}_def_dvoa",
            }
        )

    df = sched.merge(_side("home"), on=["game_id", "home_team"], how="left")
    df = df.merge(_side("away"), on=["game_id", "away_team"], how="left")

    # Simple matchup features from DVOA-style ratings
    df["off_dvoa_diff"] = df["home_off_dvoa"] - df["away_off_dvoa"]
    df["def_dvoa_diff"] = df["home_def_dvoa"] - df["away_def_dvoa"]

    # Existing flags
    df = add_home_away_flags(df)

    df.to_csv(GAME_LEVEL_FEATURES_CSV, index=False)
    print(f"Saved game level features to {GAME_LEVEL_FEATURES_CSV}")

    return df

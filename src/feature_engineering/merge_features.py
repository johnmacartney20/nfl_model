import pandas as pd

from src.utils.config import (
    RAW_DIR,
    TEAM_GAME_EPA_CSV,
    FEATURE_DIR,
    GAME_LEVEL_FEATURES_CSV,
    EXCLUDE_REG_SEASON_WEEKS_BY_SEASON,
)
from src.utils.helpers import ensure_dirs, add_home_away_flags


def build_game_level_features(seasons):
    ensure_dirs([FEATURE_DIR])

    # Load schedule
    sched_path = RAW_DIR / f"schedules_{min(seasons)}_{max(seasons)}.csv"
    sched = pd.read_csv(sched_path)

    # Optionally exclude noisy late-season regular-season weeks (e.g., rested starters)
    if EXCLUDE_REG_SEASON_WEEKS_BY_SEASON and {"season", "week", "game_type"}.issubset(set(sched.columns)):
        exclude_mask = pd.Series(False, index=sched.index)
        for season, weeks in EXCLUDE_REG_SEASON_WEEKS_BY_SEASON.items():
            if not weeks:
                continue
            exclude_mask |= (
                (sched["season"] == season)
                & (sched["game_type"] == "REG")
                & (sched["week"].isin(weeks))
            )
        sched = sched[~exclude_mask].copy()

    # Load season-to-date EPA
    STD_EPA_CSV = RAW_DIR.parent / "interim" / "season_to_date_epa.csv"
    season_epa = pd.read_csv(STD_EPA_CSV)

    # --------------------------------------------------------------------
    # Build DVOA-like strength ratings from season EPA (no external CSV)
    # --------------------------------------------------------------------
    # We assume season_epa has: season, team, team_avg_off_epa, team_avg_def_epa, team_avg_success_rate
    # Compute league averages per season
    season_epa["off_mean"] = season_epa.groupby("season")["team_avg_off_epa"].transform("mean")
    season_epa["def_mean"] = season_epa.groupby("season")["team_avg_def_epa"].transform("mean")

    # Offense "DVOA": how much better/worse than league average offense (scaled to percentage)
    season_epa["off_dvoa"] = (season_epa["team_avg_off_epa"] - season_epa["off_mean"]) * 100

    # Defense "DVOA": positive is better defense (allowing less EPA than average)
    season_epa["def_dvoa"] = -(season_epa["team_avg_def_epa"] - season_epa["def_mean"]) * 100

    # --------------------------------------------------------------------
    # Merge home team's season-to-date EPA + DVOA
    # --------------------------------------------------------------------
    df = sched.merge(
        season_epa.rename(
            columns={
                "team": "home_team",
                "team_avg_off_epa": "home_avg_off_epa",
                "team_avg_def_epa": "home_avg_def_epa",
                "team_avg_success_rate": "home_avg_success_rate",
                "off_dvoa": "home_off_dvoa",
                "def_dvoa": "home_def_dvoa",
            }
        ),
        on=["season", "home_team"],
        how="left",
    )

    # Merge away team's season-to-date EPA + DVOA
    df = df.merge(
        season_epa.rename(
            columns={
                "team": "away_team",
                "team_avg_off_epa": "away_avg_off_epa",
                "team_avg_def_epa": "away_avg_def_epa",
                "team_avg_success_rate": "away_avg_success_rate",
                "off_dvoa": "away_off_dvoa",
                "def_dvoa": "away_def_dvoa",
            }
        ),
        on=["season", "away_team"],
        how="left",
    )

    # Simple matchup features from DVOA-style ratings
    df["off_dvoa_diff"] = df["home_off_dvoa"] - df["away_off_dvoa"]
    df["def_dvoa_diff"] = df["home_def_dvoa"] - df["away_def_dvoa"]

    # Existing flags
    df = add_home_away_flags(df)

    df.to_csv(GAME_LEVEL_FEATURES_CSV, index=False)
    print(f"Saved game level features to {GAME_LEVEL_FEATURES_CSV}")

    return df

import pandas as pd
from src.utils.config import RAW_DIR, INTERIM_DIR, TEAM_GAME_EPA_CSV
from src.utils.helpers import ensure_dirs

from typing import Optional

def build_team_game_epa(pbp: Optional[pd.DataFrame] = None, seasons=None):

    """
    Build offensive and defensive EPA per play for each team in each game,
    AND season-to-date averages for each team.
    """
    ensure_dirs([INTERIM_DIR])

    if pbp is None:
        if seasons is None:
            raise ValueError("Must pass pbp or seasons")
        pbp_path = RAW_DIR / f"pbp_{min(seasons)}_{max(seasons)}.parquet"
        pbp = pd.read_parquet(pbp_path)

    # Filter to rush and pass plays
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # Per-game EPA
    off = (
        plays.groupby(["season", "game_id", "posteam"], as_index=False)
        .agg(
            off_plays=("play_id", "count"),
            off_epa=("epa", "mean"),
            off_success_rate=("success", "mean"),
        )
        .rename(columns={"posteam": "team"})
    )

    defn = (
        plays.groupby(["season", "game_id", "defteam"], as_index=False)
        .agg(
            def_plays=("play_id", "count"),
            def_epa=("epa", "mean"),
        )
        .rename(columns={"defteam": "team"})
    )

    team_game_epa = off.merge(defn, on=["season", "game_id", "team"], how="left")

    # ----- NEW: season-to-date averages -----
    season_to_date = (
        team_game_epa.groupby(["season", "team"], as_index=False)
        .agg(
            team_avg_off_epa=("off_epa", "mean"),
            team_avg_def_epa=("def_epa", "mean"),
            team_avg_success_rate=("off_success_rate", "mean"),
        )
    )
    # ----------------------------------------

    # Save outputs
    team_game_epa.to_csv(TEAM_GAME_EPA_CSV, index=False)

    STD_EPA_CSV = INTERIM_DIR / "season_to_date_epa.csv"
    season_to_date.to_csv(STD_EPA_CSV, index=False)

    print(f"Saved team game EPA to {TEAM_GAME_EPA_CSV}")
    print(f"Saved season-to-date EPA to {STD_EPA_CSV}")

    return team_game_epa, season_to_date

import pandas as pd
from src.utils.config import RAW_DIR, INTERIM_DIR, TEAM_GAME_EPA_CSV
from src.utils.config import get_excluded_reg_season_weeks_by_season
from src.utils.helpers import ensure_dirs

from typing import Optional

import numpy as np

# Point-in-time feature settings.
# Pre-game team ratings use only that team's completed games earlier in the
# same season, blended with a prior from the previous season so early weeks are
# not driven by 1-3 game samples.
PRIOR_WEIGHT_GAMES = 4.0   # prior counts as this many pseudo-games
PRIOR_REGRESSION = 0.33    # regress last season's rating this far toward league mean

PREGAME_STATS = {
    # per-game column -> output feature column
    "off_epa": "team_avg_off_epa",
    "def_epa": "team_avg_def_epa",
    "off_success_rate": "team_avg_success_rate",
}

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

    # Optionally exclude noisy late-season regular-season weeks (e.g., rested starters)
    excluded_weeks = get_excluded_reg_season_weeks_by_season()
    if excluded_weeks:
        # nfl_data_py PBP uses `season_type` (REG/POST)
        if "season" in plays.columns and "week" in plays.columns and "season_type" in plays.columns:
            exclude_mask = pd.Series(False, index=plays.index)
            for season, weeks in excluded_weeks.items():
                if not weeks:
                    continue
                exclude_mask |= (
                    (plays["season"] == season)
                    & (plays["season_type"] == "REG")
                    & (plays["week"].isin(weeks))
                )
            plays = plays[~exclude_mask].copy()

    # Per-game EPA
    off = (
        plays.groupby(["season", "week", "game_id", "posteam"], as_index=False)
        .agg(
            off_plays=("play_id", "count"),
            off_epa=("epa", "mean"),
            off_success_rate=("success", "mean"),
        )
        .rename(columns={"posteam": "team"})
    )

    defn = (
        plays.groupby(["season", "week", "game_id", "defteam"], as_index=False)
        .agg(
            def_plays=("play_id", "count"),
            def_epa=("epa", "mean"),
        )
        .rename(columns={"defteam": "team"})
    )

    team_game_epa = off.merge(defn, on=["season", "week", "game_id", "team"], how="left")

    # Full-season averages. NOTE: these include every game of the season and must
    # NOT be joined onto individual games as features (that leaks the outcome of
    # the game being predicted and later games). Game-level features use
    # compute_pregame_team_features() instead. Kept for reference/reporting.
    season_to_date = (
        team_game_epa.groupby(["season", "team"], as_index=False)
        .agg(
            team_avg_off_epa=("off_epa", "mean"),
            team_avg_def_epa=("def_epa", "mean"),
            team_avg_success_rate=("off_success_rate", "mean"),
        )
    )

    # Save outputs
    team_game_epa.to_csv(TEAM_GAME_EPA_CSV, index=False)

    STD_EPA_CSV = INTERIM_DIR / "season_to_date_epa.csv"
    season_to_date.to_csv(STD_EPA_CSV, index=False)

    print(f"Saved team game EPA to {TEAM_GAME_EPA_CSV}")
    print(f"Saved season-to-date EPA to {STD_EPA_CSV}")

    return team_game_epa, season_to_date


def _team_schedule_long(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) for every scheduled game, played or not."""
    cols = ["season", "week", "game_id"]
    order_cols = [c for c in ["gameday"] if c in sched.columns]
    home = sched[cols + order_cols + ["home_team"]].rename(columns={"home_team": "team"})
    away = sched[cols + order_cols + ["away_team"]].rename(columns={"away_team": "team"})
    long = pd.concat([home, away], ignore_index=True)
    sort_cols = ["team", "season", "week"] + order_cols + ["game_id"]
    return long.sort_values(sort_cols).reset_index(drop=True)


def _prior_from_previous_season(team_game_epa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prior for season S built only from season S-1 (fully known before S starts)."""
    full = team_game_epa.groupby(["season", "team"], as_index=False)[list(PREGAME_STATS)].mean()
    league = team_game_epa.groupby("season")[list(PREGAME_STATS)].mean()

    prior = full.copy()
    for stat in PREGAME_STATS:
        lg = prior["season"].map(league[stat])
        prior[stat] = lg + (1 - PRIOR_REGRESSION) * (prior[stat] - lg)
    prior["season"] = prior["season"] + 1  # becomes the prior for the next season

    league_prior = league.copy()
    league_prior.index = league_prior.index + 1
    return prior, league_prior


def compute_pregame_team_features(sched: pd.DataFrame, team_game_epa: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time team ratings for every scheduled (game, team).

    For each game, a team's rating uses only that team's completed games that
    come strictly earlier in the same season (by week), blended with a
    previous-season prior. The game itself and all later games are excluded.
    Unplayed games get the rating as of their kickoff, so upcoming games are
    built exactly the same way as training rows.

    Returns columns: season, week, game_id, team, team_avg_off_epa,
    team_avg_def_epa, team_avg_success_rate, team_games_played_pre,
    off_dvoa, def_dvoa.
    """
    long = _team_schedule_long(sched)
    stats = list(PREGAME_STATS)
    long = long.merge(
        team_game_epa[["season", "game_id", "team"] + stats],
        on=["season", "game_id", "team"],
        how="left",
    )

    prior, league_prior = _prior_from_previous_season(team_game_epa)
    long = long.merge(
        prior.rename(columns={s: f"{s}__prior" for s in stats}),
        on=["season", "team"],
        how="left",
    )

    # games played strictly before this row (the row's own game is excluded)
    long["_played"] = long["off_epa"].notna().astype(float)
    n_pre = long.groupby(["team", "season"], sort=False)["_played"].transform(lambda s: s.cumsum().shift(1, fill_value=0.0))
    long["team_games_played_pre"] = n_pre.astype(int)

    for stat, out in PREGAME_STATS.items():
        vals = long[stat].fillna(0.0)
        long["_v"] = vals
        sum_pre = long.groupby(["team", "season"], sort=False)["_v"].transform(
            lambda s: s.cumsum().shift(1, fill_value=0.0)
        )
        prior_val = long[f"{stat}__prior"]
        # Teams new to the data (or first season) fall back to last season's league mean.
        prior_val = prior_val.fillna(long["season"].map(league_prior[stat]))
        has_prior = prior_val.notna()
        k = np.where(has_prior, PRIOR_WEIGHT_GAMES, 0.0)
        num = sum_pre + np.where(has_prior, prior_val.fillna(0.0) * PRIOR_WEIGHT_GAMES, 0.0)
        den = n_pre + k
        long[out] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    # League reference for DVOA-style ratings: mean of all teams' pre-game
    # ratings for that week (point-in-time, no future data).
    wk = long.groupby(["season", "week"])[["team_avg_off_epa", "team_avg_def_epa"]].transform("mean")
    long["off_dvoa"] = (long["team_avg_off_epa"] - wk["team_avg_off_epa"]) * 100
    long["def_dvoa"] = -(long["team_avg_def_epa"] - wk["team_avg_def_epa"]) * 100

    keep = ["season", "week", "game_id", "team", "team_games_played_pre"] + list(PREGAME_STATS.values()) + ["off_dvoa", "def_dvoa"]
    return long[keep]

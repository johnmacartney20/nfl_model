# runs predictions for next week
from pyexpat import model
import pandas as pd
import joblib

from src.utils.config import (
    GAME_LEVEL_FEATURES_CSV,
    UPCOMING_GAMES_FEATURES_CSV,
    MODEL_PATH,
    PREDICTIONS_CSV,
    OUTPUT_DIR,
)
from src.utils.helpers import ensure_dirs

from expected_points_model import add_book_implied_scores
from sim import simulate_game_outcomes


def american_to_decimal(odds):
    if pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


def build_upcoming_games_features():
    """
    For now, "upcoming games" will be rows where scores are NaN.
    Later, you can explicitly filter by week and season.
    """
    df = pd.read_csv(GAME_LEVEL_FEATURES_CSV)
    upcoming = df[df["home_score"].isna() | df["away_score"].isna()].copy()

    upcoming.to_csv(UPCOMING_GAMES_FEATURES_CSV, index=False)
    print(f"Saved upcoming features to {UPCOMING_GAMES_FEATURES_CSV}")
    return upcoming


def predict_upcoming_games():
    ensure_dirs([OUTPUT_DIR])

    clf = joblib.load(MODEL_PATH)

    # Build the upcoming games table
    upcoming = build_upcoming_games_features()

    feature_cols = [
    "home_avg_off_epa",
    "home_avg_def_epa",
    "home_avg_success_rate",
    "away_avg_off_epa",
    "away_avg_def_epa",
    "away_avg_success_rate",
    "home_off_dvoa",
    "home_def_dvoa",
    "away_off_dvoa",
    "away_def_dvoa",
    "off_dvoa_diff",
    "def_dvoa_diff",
]


    # Remove any rows missing features
    upcoming = upcoming.dropna(subset=feature_cols)

    # ----- SAFE GUARD -----
    if upcoming.empty:
        print("No upcoming games found. Skipping predictions.")
        return
    # -----------------------

    # Make predictions (home win probability from logistic model)
    X = upcoming[feature_cols]
    probs = clf.predict_proba(X)[:, 1]
    upcoming["home_win_prob"] = probs

    # ---- Book implied scores from spread + total ----
    # upcoming already has: spread_line, total_line, over_odds, under_odds, home_team, away_team, etc.
    games_df = add_book_implied_scores(
        upcoming,
        spread_col="spread_line",
        total_col="total_line",
    )

    # ---- Model expected scores based on EPA and DVOA ----
    # League average points per game (roughly 22-23 points per team)
    league_avg_score = 22.5
    home_field_advantage = 2.5
    
    # Scale EPA to points (EPA is typically between -0.2 and +0.2 for good/bad teams)
    # A team with +0.1 EPA should score about 2-3 points more than average
    epa_to_points_scale = 25.0
    
    # Calculate model scores based on offensive/defensive strength
    # Home team score = base + home advantage + offensive strength - opponent defensive strength
    games_df["model_home_score"] = (
        league_avg_score 
        + home_field_advantage
        + (games_df["home_avg_off_epa"] * epa_to_points_scale)
        - (games_df["away_avg_def_epa"] * epa_to_points_scale)
    )
    
    # Away team score = base + offensive strength - opponent defensive strength
    games_df["model_away_score"] = (
        league_avg_score
        + (games_df["away_avg_off_epa"] * epa_to_points_scale)
        - (games_df["home_avg_def_epa"] * epa_to_points_scale)
    )

    # ---- Run simulation to get win / cover / over probabilities ----
    games_df = simulate_game_outcomes(games_df, n_sims=5000, sigma=10.0)

    # Now model_over_pct exists, so we can safely use it
    games_df["model_under_pct"] = 1 - games_df["model_over_pct"]

    # Model expected total from "model" scores
    games_df["model_total_points"] = (
        games_df["model_home_score"] + games_df["model_away_score"]
    )

    # Treat missing odds as NaN (books might store them as 0)
    for col in ["over_odds", "under_odds"]:
        games_df.loc[games_df[col] == 0, col] = pd.NA


    # Convert American odds to decimal odds for totals
    games_df["dec_over_odds"] = games_df["over_odds"].apply(american_to_decimal)
    games_df["dec_under_odds"] = games_df["under_odds"].apply(american_to_decimal)

    # Expected value per $1 on over / under
    games_df["ev_over_per_$"] = (
        games_df["model_over_pct"] * (games_df["dec_over_odds"] - 1)
        - (1 - games_df["model_over_pct"])
    )

    games_df["ev_under_per_$"] = (
        games_df["model_under_pct"] * (games_df["dec_under_odds"] - 1)
        - (1 - games_df["model_under_pct"])
    )

    # Save full predictions with simulation output
    games_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved predictions (with simulation) to {PREDICTIONS_CSV}")

    # Print a quick view for sanity check
    cols_to_show = [
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "book_home_score",
        "book_away_score",
        "model_total_points",
        "home_win_prob",
        "model_win_pct_home",
        "model_cover_pct_home",
        "model_over_pct",
        "model_under_pct",
        "over_odds",
        "under_odds",
        "ev_over_per_$",
        "ev_under_per_$",
    ]
    cols_to_show = [c for c in cols_to_show if c in games_df.columns]
    print(games_df[cols_to_show])

    return games_df


if __name__ == "__main__":
    predict_upcoming_games()

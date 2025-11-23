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

    model = joblib.load(MODEL_PATH)

    # Build the upcoming games table
    upcoming = build_upcoming_games_features()

    feature_cols = [
    "home_avg_off_epa",
    "home_avg_def_epa",
    "home_avg_success_rate",
    "away_avg_off_epa",
    "away_avg_def_epa",
    "away_avg_success_rate",
    ]

    # Remove any rows missing features
    upcoming = upcoming.dropna(subset=feature_cols)

    # ----- SAFE GUARD -----
    if upcoming.empty:
        print("No upcoming games found. Skipping predictions.")
        return
    # -----------------------

    # Make predictions
    X = upcoming[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    # Save predictions
    upcoming["home_win_prob"] = probs
    upcoming.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved predictions to {PREDICTIONS_CSV}")

    return upcoming

import pandas as pd
from sklearn.linear_model import Ridge


def predict_scores_from_epa(games: pd.DataFrame) -> pd.DataFrame:
    """
    Predict scores using EPA-based model instead of just reversing Vegas lines.
    
    Uses offensive and defensive EPA to predict scoring, with adjustments for:
    - Home field advantage (~2.5 points)
    - League average scoring
    
    Calibrated to match actual 2025 season average of 46.2 ppg (23.1 per team).
    """
    # League baseline (2025 season average is 46.2 ppg total, so 23.1 per team)
    league_avg = 23.1
    home_field_advantage = 2.2  # Reduced from 2.5
    
    # EPA scaling factor - aggressively reduced to fix +4.0 bias
    # Was 25.0 (too high), then 22.0 (still too high)
    epa_to_points = 18.0
    
    # Calculate expected scoring for each team
    # Home team score = league avg + home advantage + offensive EPA effect - opponent defensive EPA effect
    games['model_home_score'] = (
        league_avg + 
        home_field_advantage +
        games['home_avg_off_epa'] * epa_to_points -
        games['away_avg_def_epa'] * epa_to_points
    )
    
    # Away team score = league avg + offensive EPA effect - opponent defensive EPA effect
    games['model_away_score'] = (
        league_avg +
        games['away_avg_off_epa'] * epa_to_points -
        games['home_avg_def_epa'] * epa_to_points
    )
    
    return games


def add_book_implied_scores(
    games: pd.DataFrame,
    spread_col: str = "spread_line",
    total_col: str = "total_line",
) -> pd.DataFrame:
    """
    Compute book implied scores from spread + total.
    
    NOTE: This is kept for backwards compatibility but predict_scores_from_epa 
    should be used instead for actual predictions.

    spread_col: spread from home team perspective.
                Example: home -3 means -3 here.
    total_col: game total points line.
    """
    spread = games[spread_col]
    total = games[total_col]

    games["book_home_score"] = (total - spread) / 2
    games["book_away_score"] = (total + spread) / 2
    return games


class ExpectedPointsModel:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_cols = [
            "off_rating_team",
            "def_rating_opp",
            "home_flag",
        ]
        self.is_fitted = False

    def fit(self, df_hist: pd.DataFrame):
        X = df_hist[self.feature_cols]
        y = df_hist["points_scored"]
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_points(self, df_games: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("ExpectedPointsModel is not fitted yet")

        X = df_games[self.feature_cols]
        return pd.Series(self.model.predict(X), index=df_games.index)

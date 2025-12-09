import pandas as pd
from sklearn.linear_model import Ridge


def add_book_implied_scores(
    games: pd.DataFrame,
    spread_col: str = "spread_line",
    total_col: str = "total_line",
) -> pd.DataFrame:
    """
    Compute book implied scores from spread + total.

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

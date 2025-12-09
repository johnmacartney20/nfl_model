import pandas as pd
import numpy as np


def simulate_game_outcomes(
    games: pd.DataFrame,
    n_sims: int = 5000,
    sigma: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate game outcomes based on expected scores.

    Requires columns:
      model_home_score
      model_away_score
      spread_line      (home spread, eg -3.5 if home is favoured by 3.5)
      total_line       (game total points line)

    Adds columns:
      model_win_pct_home
      model_cover_pct_home
      model_over_pct
    """
    results = []

    for _, row in games.iterrows():
        mu_home = row["model_home_score"]
        mu_away = row["model_away_score"]
        spread = row["spread_line"]
        total_line = row["total_line"]

        # Simulate scores
        home_scores = np.random.normal(mu_home, sigma, size=n_sims)
        away_scores = np.random.normal(mu_away, sigma, size=n_sims)

        home_scores = np.clip(home_scores, 0, None)
        away_scores = np.clip(away_scores, 0, None)

        margin = home_scores - away_scores
        totals = home_scores + away_scores

        # Home straight up win
        win_home = (margin > 0).mean()

        # Home ATS:
        # If spread_line is -3.5 for home, they cover when margin + spread_line > 0
        cover_home = (margin + spread > 0).mean()

        # Over total
        over_prob = (totals > total_line).mean()

        results.append(
            {
                "model_win_pct_home": win_home,
                "model_cover_pct_home": cover_home,
                "model_over_pct": over_prob,
            }
        )

    sim_df = pd.DataFrame(results)
    sim_df.index = games.index  # align row order
    return pd.concat([games, sim_df], axis=1)

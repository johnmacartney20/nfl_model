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


def calculate_kelly_fraction(probability, decimal_odds, kelly_fraction=0.25):
    """
    Calculate Kelly Criterion bet size.
    
    Args:
        probability: Model's estimated probability of winning the bet
        decimal_odds: Decimal odds for the bet
        kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly for safety)
    
    Returns:
        Recommended fraction of bankroll to bet (0 if no edge)
    """
    if pd.isna(probability) or pd.isna(decimal_odds):
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # where b = decimal_odds - 1, p = probability, q = 1 - p
    b = decimal_odds - 1
    p = probability
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Only bet if there's positive expected value
    if kelly <= 0:
        return 0.0
    
    # Use fractional Kelly for safety (reduces variance)
    return kelly * kelly_fraction


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

    # ---- Calculate probabilities analytically (faster and more stable than simulation) ----
    # Calculate probabilities using logistic model for consistency
    # Logistic regression performed better (64% accuracy vs 36% for score-based analytical)
    from scipy import stats
    
    sigma_margin = 13.5  # Standard deviation of point margin in NFL
    sigma_total = 10.5   # Standard deviation of total points in NFL
    
    def calculate_all_probs(row):
        # Use logistic win probability as the base
        # Convert to expected margin via inverse normal CDF
        # If P(win) = P(margin > 0), then margin ~ N(μ, σ) where P(Z > -μ/σ) = P(win)
        # So μ = -σ * Z where Z is the z-score corresponding to P(win)
        win_prob = row["home_win_prob"]
        
        # Find the expected margin that produces this win probability
        # norm.ppf gives us the z-score, multiply by sigma to get expected margin
        expected_margin = stats.norm.ppf(win_prob) * sigma_margin
        
        # Cover probability: home covers if (margin + spread_line) > 0
        # Rearranged: margin > -spread_line
        cover_threshold = -row["spread_line"]
        cover_prob = 1 - stats.norm.cdf(cover_threshold, loc=expected_margin, scale=sigma_margin)
        
        # Over/under uses score predictions (independent from win/cover)
        expected_total = row["model_home_score"] + row["model_away_score"]
        over_prob = 1 - stats.norm.cdf(row["total_line"], loc=expected_total, scale=sigma_total)
        
        return pd.Series({
            'model_cover_pct_home': cover_prob,
            'model_over_pct': over_prob
        })
    
    analytical_probs = games_df.apply(calculate_all_probs, axis=1)
    
    # Use logistic regression for all margin-related bets (moneyline and spreads)
    games_df["model_win_pct_home"] = games_df["home_win_prob"]  # Logistic regression
    games_df["model_win_pct_away"] = 1 - games_df["home_win_prob"]
    games_df["model_cover_pct_home"] = analytical_probs["model_cover_pct_home"]
    games_df["model_over_pct"] = analytical_probs["model_over_pct"]
    games_df["model_under_pct"] = 1 - games_df["model_over_pct"]

    # Model expected total from "model" scores
    games_df["model_total_points"] = (
        games_df["model_home_score"] + games_df["model_away_score"]
    )

    # Treat missing odds as NaN (books might store them as 0)
    for col in ["over_odds", "under_odds"]:
        games_df.loc[games_df[col] == 0, col] = pd.NA


    # Convert American odds to decimal odds
    games_df["dec_over_odds"] = games_df["over_odds"].apply(american_to_decimal)
    games_df["dec_under_odds"] = games_df["under_odds"].apply(american_to_decimal)
    
    # Also get spread odds
    games_df["dec_home_spread_odds"] = games_df["home_spread_odds"].apply(american_to_decimal)
    games_df["dec_away_spread_odds"] = games_df["away_spread_odds"].apply(american_to_decimal)
    
    # Moneyline odds
    games_df["dec_home_moneyline"] = games_df["home_moneyline"].apply(american_to_decimal)
    games_df["dec_away_moneyline"] = games_df["away_moneyline"].apply(american_to_decimal)

    # ============================================================
    # EXPECTED VALUE CALCULATIONS
    # ============================================================
    
    # Over/Under Expected Value
    games_df["ev_over"] = (
        games_df["model_over_pct"] * (games_df["dec_over_odds"] - 1)
        - (1 - games_df["model_over_pct"])
    )
    games_df["ev_under"] = (
        games_df["model_under_pct"] * (games_df["dec_under_odds"] - 1)
        - (1 - games_df["model_under_pct"])
    )
    
    # Spread Expected Value
    games_df["ev_home_spread"] = (
        games_df["model_cover_pct_home"] * (games_df["dec_home_spread_odds"] - 1)
        - (1 - games_df["model_cover_pct_home"])
    )
    games_df["ev_away_spread"] = (
        (1 - games_df["model_cover_pct_home"]) * (games_df["dec_away_spread_odds"] - 1)
        - games_df["model_cover_pct_home"]
    )
    
    # Moneyline Expected Value
    games_df["ev_home_ml"] = (
        games_df["model_win_pct_home"] * (games_df["dec_home_moneyline"] - 1)
        - (1 - games_df["model_win_pct_home"])
    )
    games_df["ev_away_ml"] = (
        (1 - games_df["model_win_pct_home"]) * (games_df["dec_away_moneyline"] - 1)
        - games_df["model_win_pct_home"]
    )

    # ============================================================
    # KELLY CRITERION BET SIZING (using quarter-Kelly for safety)
    # ============================================================
    
    # Over/Under Kelly
    games_df["kelly_over"] = games_df.apply(
        lambda row: calculate_kelly_fraction(row["model_over_pct"], row["dec_over_odds"]), 
        axis=1
    )
    games_df["kelly_under"] = games_df.apply(
        lambda row: calculate_kelly_fraction(row["model_under_pct"], row["dec_under_odds"]), 
        axis=1
    )
    
    # Spread Kelly
    games_df["kelly_home_spread"] = games_df.apply(
        lambda row: calculate_kelly_fraction(row["model_cover_pct_home"], row["dec_home_spread_odds"]), 
        axis=1
    )
    games_df["kelly_away_spread"] = games_df.apply(
        lambda row: calculate_kelly_fraction(1 - row["model_cover_pct_home"], row["dec_away_spread_odds"]), 
        axis=1
    )
    
    # Moneyline Kelly
    games_df["kelly_home_ml"] = games_df.apply(
        lambda row: calculate_kelly_fraction(row["model_win_pct_home"], row["dec_home_moneyline"]), 
        axis=1
    )
    games_df["kelly_away_ml"] = games_df.apply(
        lambda row: calculate_kelly_fraction(1 - row["model_win_pct_home"], row["dec_away_moneyline"]), 
        axis=1
    )
    
    # ============================================================
    # IDENTIFY BEST BETS (highest expected value with positive Kelly)
    # ============================================================
    
    # Find the best bet for each game
    bet_columns = {
        'over': ('ev_over', 'kelly_over'),
        'under': ('ev_under', 'kelly_under'),
        'home_spread': ('ev_home_spread', 'kelly_home_spread'),
        'away_spread': ('ev_away_spread', 'kelly_away_spread'),
        'home_ml': ('ev_home_ml', 'kelly_home_ml'),
        'away_ml': ('ev_away_ml', 'kelly_away_ml')
    }
    
    def find_best_bet(row):
        best_ev = -1
        best_bet = None
        best_kelly = 0
        
        for bet_name, (ev_col, kelly_col) in bet_columns.items():
            ev = row[ev_col]
            kelly = row[kelly_col]
            if pd.notna(ev) and pd.notna(kelly) and ev > best_ev and kelly > 0:
                best_ev = ev
                best_bet = bet_name
                best_kelly = kelly
        
        return pd.Series({
            'best_bet_type': best_bet if best_bet else 'no_edge',
            'best_bet_ev': best_ev if best_bet else 0,
            'best_bet_kelly': best_kelly
        })
    
    best_bets_df = games_df.apply(find_best_bet, axis=1)
    games_df = pd.concat([games_df, best_bets_df], axis=1)

    # Save full predictions with simulation output
    games_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved predictions (with simulation) to {PREDICTIONS_CSV}")

    # Print a summary of best bets
    print("\n" + "=" * 120)
    print("TOP BETTING OPPORTUNITIES (Highest Expected Value)")
    print("=" * 120)
    
    # Filter to games with positive EV and valid team names
    value_bets = games_df[
        (games_df['best_bet_ev'] > 0) & 
        (games_df['home_team'].notna()) & 
        (games_df['away_team'].notna())
    ].copy()
    
    if not value_bets.empty:
        value_bets = value_bets.sort_values('best_bet_ev', ascending=False)
        
        display_cols = [
            'home_team', 'away_team', 'best_bet_type', 
            'best_bet_ev', 'best_bet_kelly'
        ]
        print(value_bets[display_cols].head(10).to_string(index=False))
        print(f"\nTotal games with positive EV: {len(value_bets)}")
        print(f"Average Kelly bet size on value bets: {value_bets['best_bet_kelly'].mean():.2%} of bankroll")
    else:
        print("No positive expected value bets found.")
    
    # Print detailed breakdown for games with valid data
    print("\n" + "=" * 120)
    print("ALL BETS - EXPECTED VALUE & KELLY CRITERION")
    print("=" * 120)
    
    valid_games = games_df[games_df['home_team'].notna() & games_df['away_team'].notna()].copy()
    
    if not valid_games.empty:
        # Show all betting options with EV and Kelly
        print("\nOver/Under Bets:")
        print("-" * 120)
        ou_cols = ["home_team", "away_team", "total_line", "model_total_points", "ev_over", "kelly_over", "ev_under", "kelly_under"]
        ou_cols = [c for c in ou_cols if c in valid_games.columns]
        print(valid_games[ou_cols].to_string(index=False))
        
        print("\n\nSpread Bets:")
        print("-" * 120)
        spread_cols = ["home_team", "away_team", "spread_line", "model_cover_pct_home", "ev_home_spread", "kelly_home_spread", "ev_away_spread", "kelly_away_spread"]
        spread_cols = [c for c in spread_cols if c in valid_games.columns]
        print(valid_games[spread_cols].head(10).to_string(index=False))
        
        print("\n\nMoneyline Bets:")
        print("-" * 120)
        ml_cols = ["home_team", "away_team", "model_win_pct_home", "home_moneyline", "ev_home_ml", "kelly_home_ml", "away_moneyline", "ev_away_ml", "kelly_away_ml"]
        ml_cols = [c for c in ml_cols if c in valid_games.columns]
        print(valid_games[ml_cols].head(10).to_string(index=False))
        
        print("\n\nBest Bet Summary:")
        print("-" * 120)
        summary_cols = ["home_team", "away_team", "best_bet_type", "best_bet_ev", "best_bet_kelly"]
        summary_cols = [c for c in summary_cols if c in valid_games.columns]
        print(valid_games[summary_cols].to_string(index=False))

    return games_df


if __name__ == "__main__":
    predict_upcoming_games()

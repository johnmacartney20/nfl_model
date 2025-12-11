"""
Evaluate model performance on completed games from past weeks.
This shows what predictions the model would have made before games were played.
"""
import pandas as pd
import joblib
import sys

from src.utils.config import GAME_LEVEL_FEATURES_CSV, MODEL_PATH
from expected_points_model import add_book_implied_scores, predict_scores_from_epa
from sim import simulate_game_outcomes


def evaluate_past_week(season=2025, week=None):
    """
    Generate retroactive predictions for a past week and compare to actual results.
    
    Args:
        season: NFL season year
        week: Week number. If None, uses most recent completed week.
    
    Returns:
        DataFrame with predictions and actual results
    """
    # Load model
    clf = joblib.load(MODEL_PATH)
    
    # Load all games
    df = pd.read_csv(GAME_LEVEL_FEATURES_CSV)
    
    # If week not specified, find most recent completed week
    if week is None:
        completed = df[(df['season'] == season) & df['home_score'].notna()]
        if completed.empty:
            print(f"No completed games found for season {season}")
            return None
        week = completed['week'].max()
        print(f"Using most recent completed week: {week}\n")
    
    # Get games for specified week
    week_games = df[(df['season'] == season) & (df['week'] == week)].copy()
    
    # Filter to only completed games
    week_games = week_games[week_games['home_score'].notna()].copy()
    
    if week_games.empty:
        print(f"No completed games found for season {season}, week {week}")
        return None
    
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
        "div_game",
        "home_rest",
        "away_rest",
    ]
    
    # Remove rows missing features
    week_games = week_games.dropna(subset=feature_cols)
    
    if week_games.empty:
        print(f"No games with complete features for season {season}, week {week}")
        return None
    
    # Predict home win probability
    X = week_games[feature_cols]
    probs = clf.predict_proba(X)[:, 1]
    week_games["home_win_prob"] = probs
    
    # Model expected scores using EPA-based predictions
    week_games = predict_scores_from_epa(week_games)
    
    # Also add book implied scores for reference
    week_games = add_book_implied_scores(
        week_games, 
        spread_col="spread_line", 
        total_col="total_line"
    )
    
    # Run simulation
    week_games = simulate_game_outcomes(week_games, n_sims=5000, sigma=10.0)
    week_games["model_total_points"] = week_games["model_home_score"] + week_games["model_away_score"]
    
    # Calculate actual results
    week_games["actual_total"] = week_games["home_score"] + week_games["away_score"]
    week_games["actual_margin"] = week_games["home_score"] - week_games["away_score"]
    week_games["home_won"] = (week_games["home_score"] > week_games["away_score"]).astype(int)
    week_games["home_covered"] = (week_games["actual_margin"] > week_games["spread_line"]).astype(int)
    week_games["actual_over"] = (week_games["actual_total"] > week_games["total_line"]).astype(int)
    
    # Model predictions
    week_games["model_predicted_home_win"] = (week_games["home_win_prob"] > 0.5).astype(int)
    week_games["model_predicted_cover"] = (week_games["model_cover_pct_home"] > 0.5).astype(int)
    week_games["model_predicted_over"] = (week_games["model_over_pct"] > 0.5).astype(int)
    
    # Calculate accuracy
    week_games["win_correct"] = (week_games["home_won"] == week_games["model_predicted_home_win"]).astype(int)
    week_games["cover_correct"] = (week_games["home_covered"] == week_games["model_predicted_cover"]).astype(int)
    week_games["ou_correct"] = (week_games["actual_over"] == week_games["model_predicted_over"]).astype(int)
    
    return week_games


def print_results(week_games, season, week):
    """Print formatted results for a week."""
    
    cols = [
        "home_team", "away_team", 
        "home_score", "away_score", "actual_total",
        "model_home_score", "model_away_score", "model_total_points",
        "book_home_score", "book_away_score",
        "spread_line", "total_line",
        "home_win_prob", "home_won",
        "model_cover_pct_home", "home_covered",
        "model_over_pct", "actual_over"
    ]
    
    # Only show columns that exist
    cols = [c for c in cols if c in week_games.columns]
    
    print("=" * 120)
    print(f"WEEK {week} ({season}) - RETROACTIVE PREDICTIONS vs ACTUAL RESULTS")
    print("=" * 120)
    print(week_games[cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 120)
    
    n_games = len(week_games)
    win_correct = week_games["win_correct"].sum()
    cover_correct = week_games["cover_correct"].sum()
    ou_correct = week_games["ou_correct"].sum()
    
    print(f"Games Analyzed: {n_games}")
    print(f"\nWin Predictions: {win_correct}/{n_games} correct ({100*win_correct/n_games:.1f}%)")
    print(f"Cover Predictions: {cover_correct}/{n_games} correct ({100*cover_correct/n_games:.1f}%)")
    print(f"Over/Under Predictions: {ou_correct}/{n_games} correct ({100*ou_correct/n_games:.1f}%)")
    
    # Scoring accuracy
    model_error = abs(week_games['model_total_points'] - week_games['actual_total']).mean()
    book_error = abs((week_games['book_home_score'] + week_games['book_away_score']) - week_games['actual_total']).mean()
    
    print(f"\nAverage Total Points Error:")
    print(f"  Model: {model_error:.2f} points")
    print(f"  Book:  {book_error:.2f} points")
    
    # Show best predictions (smallest errors)
    print("\n" + "=" * 120)
    print("BEST PREDICTIONS (Smallest Total Points Error)")
    print("=" * 120)
    week_games_copy = week_games.copy()
    week_games_copy['total_error'] = abs(week_games_copy['model_total_points'] - week_games_copy['actual_total'])
    best = week_games_copy.nsmallest(5, 'total_error')[
        ['home_team', 'away_team', 'actual_total', 'model_total_points', 'total_error']
    ]
    print(best.to_string(index=False))
    
    # Show worst predictions (largest errors)
    print("\n" + "=" * 120)
    print("WORST PREDICTIONS (Largest Total Points Error)")
    print("=" * 120)
    worst = week_games_copy.nlargest(5, 'total_error')[
        ['home_team', 'away_team', 'actual_total', 'model_total_points', 'total_error']
    ]
    print(worst.to_string(index=False))


def analyze_multiple_weeks(season=2025, start_week=1, end_week=None):
    """
    Analyze model performance across multiple weeks.
    
    Args:
        season: NFL season year
        start_week: First week to analyze
        end_week: Last week to analyze (None = most recent completed)
    
    Returns:
        DataFrame with aggregated statistics
    """
    df = pd.read_csv(GAME_LEVEL_FEATURES_CSV)
    
    if end_week is None:
        completed = df[(df['season'] == season) & df['home_score'].notna()]
        end_week = completed['week'].max() if not completed.empty else start_week
    
    all_results = []
    
    for week in range(start_week, end_week + 1):
        result = evaluate_past_week(season=season, week=week)
        if result is not None and not result.empty:
            result['week'] = week
            all_results.append(result)
    
    if not all_results:
        print("No results found for specified weeks")
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Overall statistics
    print("\n" + "=" * 120)
    print(f"OVERALL PERFORMANCE: Weeks {start_week}-{end_week} ({season})")
    print("=" * 120)
    
    n_games = len(combined)
    win_correct = combined["win_correct"].sum()
    cover_correct = combined["cover_correct"].sum()
    ou_correct = combined["ou_correct"].sum()
    
    print(f"Total Games: {n_games}")
    print(f"\nWin Predictions: {win_correct}/{n_games} correct ({100*win_correct/n_games:.1f}%)")
    print(f"Cover Predictions: {cover_correct}/{n_games} correct ({100*cover_correct/n_games:.1f}%)")
    print(f"Over/Under Predictions: {ou_correct}/{n_games} correct ({100*ou_correct/n_games:.1f}%)")
    
    model_error = abs(combined['model_total_points'] - combined['actual_total']).mean()
    book_error = abs((combined['book_home_score'] + combined['book_away_score']) - combined['actual_total']).mean()
    
    print(f"\nAverage Total Points Error:")
    print(f"  Model: {model_error:.2f} points")
    print(f"  Book:  {book_error:.2f} points")
    
    # Week by week breakdown
    print("\n" + "=" * 120)
    print("WEEK-BY-WEEK BREAKDOWN")
    print("=" * 120)
    
    weekly_stats = combined.groupby('week').agg({
        'win_correct': 'sum',
        'cover_correct': 'sum',
        'ou_correct': 'sum',
        'home_team': 'count'
    }).rename(columns={'home_team': 'games'})
    
    weekly_stats['win_pct'] = (100 * weekly_stats['win_correct'] / weekly_stats['games']).round(1)
    weekly_stats['cover_pct'] = (100 * weekly_stats['cover_correct'] / weekly_stats['games']).round(1)
    weekly_stats['ou_pct'] = (100 * weekly_stats['ou_correct'] / weekly_stats['games']).round(1)
    
    print(weekly_stats[['games', 'win_correct', 'win_pct', 'cover_correct', 'cover_pct', 'ou_correct', 'ou_pct']].to_string())
    
    return combined


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # Analyze all completed weeks
            analyze_multiple_weeks(season=2025)
        else:
            # Single week specified
            week = int(sys.argv[1])
            season = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
            result = evaluate_past_week(season=season, week=week)
            if result is not None:
                print_results(result, season, week)
    else:
        # Default: analyze most recent week
        result = evaluate_past_week(season=2025)
        if result is not None:
            week = result['week'].iloc[0]
            print_results(result, 2025, week)

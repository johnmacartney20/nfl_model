"""
Direct probability calculations without Monte Carlo simulation.
Uses analytical formulas assuming normal distribution of scores.
"""
import numpy as np
from scipy import stats


def calculate_probabilities_analytical(
    home_score: float,
    away_score: float,
    spread_line: float,
    total_line: float,
    sigma: float = 13.5  # Standard deviation of score margin in NFL
) -> dict:
    """
    Calculate win/cover/over probabilities using analytical formulas.
    
    Args:
        home_score: Expected home team score
        away_score: Expected away team score
        spread_line: Spread line (negative = home favored)
        total_line: Total points line
        sigma: Standard deviation of score margin (default ~13.5 for NFL)
    
    Returns:
        Dictionary with probabilities for win, cover, over
    """
    # Expected margin (home - away)
    expected_margin = home_score - away_score
    expected_total = home_score + away_score
    
    # Win probability (home wins straight up)
    # P(margin > 0)
    win_prob = 1 - stats.norm.cdf(0, loc=expected_margin, scale=sigma)
    
    # Cover probability (home covers the spread)
    # If spread_line = -3.5, home needs to win by more than 3.5
    # P(margin + spread_line > 0) = P(margin > -spread_line)
    cover_threshold = -spread_line
    cover_prob = 1 - stats.norm.cdf(cover_threshold, loc=expected_margin, scale=sigma)
    
    # Over probability
    # Total points have higher variance (sigma_total ≈ sigma_margin * sqrt(2))
    # But we can use empirical sigma for totals (~10-11 points)
    sigma_total = 10.5  # Empirical standard deviation for NFL totals
    over_prob = 1 - stats.norm.cdf(total_line, loc=expected_total, scale=sigma_total)
    
    return {
        'win_prob': win_prob,
        'cover_prob': cover_prob,
        'over_prob': over_prob,
        'under_prob': 1 - over_prob,
        'expected_margin': expected_margin,
        'expected_total': expected_total
    }


def calculate_all_bet_probabilities(
    home_score: float,
    away_score: float,
    spread_line: float,
    total_line: float,
    sigma_margin: float = 13.5,
    sigma_total: float = 10.5
) -> dict:
    """
    Calculate all betting probabilities with configurable sigma values.
    
    This uses the analytical approach which is:
    - Much faster than simulation (no Monte Carlo needed)
    - More stable (no random variance)
    - Theoretically sound (assumes normal distribution)
    
    NFL empirical values:
    - sigma_margin ≈ 13.5 points (standard deviation of point margin)
    - sigma_total ≈ 10.5 points (standard deviation of total points)
    
    Returns:
        Dictionary with all probabilities and expected values
    """
    expected_margin = home_score - away_score
    expected_total = home_score + away_score
    
    # Straight up win probability
    win_prob_home = 1 - stats.norm.cdf(0, loc=expected_margin, scale=sigma_margin)
    win_prob_away = 1 - win_prob_home
    
    # Against the spread probabilities
    # Home covers if: (home_score - away_score) > -spread_line
    # Example: spread_line = -3.5 means home favored by 3.5
    #          Home covers if margin > 3.5
    cover_threshold = -spread_line
    cover_prob_home = 1 - stats.norm.cdf(cover_threshold, loc=expected_margin, scale=sigma_margin)
    cover_prob_away = 1 - cover_prob_home
    
    # Over/Under probabilities
    over_prob = 1 - stats.norm.cdf(total_line, loc=expected_total, scale=sigma_total)
    under_prob = 1 - over_prob
    
    return {
        # Win probabilities
        'win_prob_home': win_prob_home,
        'win_prob_away': win_prob_away,
        
        # Spread probabilities
        'cover_prob_home': cover_prob_home,
        'cover_prob_away': cover_prob_away,
        
        # Total probabilities
        'over_prob': over_prob,
        'under_prob': under_prob,
        
        # Expected values
        'expected_margin': expected_margin,
        'expected_total': expected_total,
        
        # Push probabilities (for reference, though rare with half-point lines)
        'push_prob_spread': stats.norm.pdf(cover_threshold, loc=expected_margin, scale=sigma_margin) * 0.5,
        'push_prob_total': stats.norm.pdf(total_line, loc=expected_total, scale=sigma_total) * 0.5
    }


# Tuning parameters based on historical data
def estimate_sigma_from_historical_data(df, margin_col='actual_margin', total_col='actual_total'):
    """
    Estimate optimal sigma values from historical game data.
    
    Args:
        df: DataFrame with actual game results
        margin_col: Column name for point margin (home - away)
        total_col: Column name for total points
    
    Returns:
        Dictionary with optimal sigma values
    """
    sigma_margin = df[margin_col].std()
    sigma_total = df[total_col].std()
    
    return {
        'sigma_margin': sigma_margin,
        'sigma_total': sigma_total
    }

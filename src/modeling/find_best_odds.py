"""
Compare your model's predictions to odds across multiple sportsbooks.
Identifies the best available odds for your value bets.
"""
import pandas as pd
from src.data_ingest.import_odds import fetch_nfl_odds, compare_to_fanduel
from src.modeling.predict_upcoming import predict_upcoming_games, american_to_decimal, calculate_kelly_fraction


# Team name mappings (The Odds API uses different abbreviations)
ODDS_API_TO_NFL = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
}


def find_best_moneyline_opportunities(predictions_df, odds_df):
    """
    Match model predictions with sportsbook odds to find best opportunities.
    
    Args:
        predictions_df: Your model's predictions (from predict_upcoming_games)
        odds_df: Odds from multiple books (from fetch_nfl_odds)
    
    Returns:
        DataFrame with best odds for each positive EV bet
    """
    # Filter to moneyline odds
    ml_odds = odds_df[odds_df['market'] == 'moneyline'].copy()
    
    # Map team names
    ml_odds['team_abbr'] = ml_odds['team'].map(ODDS_API_TO_NFL)
    ml_odds['home_team_abbr'] = ml_odds['home_team'].map(ODDS_API_TO_NFL)
    ml_odds['away_team_abbr'] = ml_odds['away_team'].map(ODDS_API_TO_NFL)
    
    opportunities = []
    
    for _, pred in predictions_df.iterrows():
        home_team = pred['home_team']
        away_team = pred['away_team']
        
        # Get all odds for this game
        game_odds = ml_odds[
            (ml_odds['home_team_abbr'] == home_team) & 
            (ml_odds['away_team_abbr'] == away_team)
        ]
        
        if game_odds.empty:
            continue
        
        # Check home team moneyline
        home_win_prob = pred.get('model_win_pct_home', pred.get('home_win_prob'))
        home_odds_available = game_odds[game_odds['team_abbr'] == home_team]
        
        if not home_odds_available.empty:
            # Find best odds (highest value = best payout)
            best_home = home_odds_available.loc[home_odds_available['odds'].idxmax()]
            
            # Calculate EV with best odds
            best_decimal = american_to_decimal(best_home['odds'])
            ev = (home_win_prob * (best_decimal - 1)) - ((1 - home_win_prob) * 1)
            kelly = calculate_kelly_fraction(home_win_prob, best_decimal, bet_type='moneyline') * 100
            
            if ev > 0:  # Positive EV
                opportunities.append({
                    'team': home_team,
                    'opponent': away_team,
                    'location': 'home',
                    'model_prob': home_win_prob,
                    'best_odds': best_home['odds'],
                    'best_bookmaker': best_home['bookmaker'],
                    'decimal_odds': best_decimal,
                    'ev': ev,
                    'kelly_bet': kelly,
                    'fanduel_odds': home_odds_available[
                        home_odds_available['bookmaker'] == 'fanduel'
                    ]['odds'].iloc[0] if 'fanduel' in home_odds_available['bookmaker'].values else None
                })
        
        # Check away team moneyline
        away_win_prob = 1 - home_win_prob
        away_odds_available = game_odds[game_odds['team_abbr'] == away_team]
        
        if not away_odds_available.empty:
            best_away = away_odds_available.loc[away_odds_available['odds'].idxmax()]
            
            best_decimal = american_to_decimal(best_away['odds'])
            ev = (away_win_prob * (best_decimal - 1)) - ((1 - away_win_prob) * 1)
            kelly = calculate_kelly_fraction(away_win_prob, best_decimal, bet_type='moneyline') * 100
            
            if ev > 0:
                opportunities.append({
                    'team': away_team,
                    'opponent': home_team,
                    'location': 'away',
                    'model_prob': away_win_prob,
                    'best_odds': best_away['odds'],
                    'best_bookmaker': best_away['bookmaker'],
                    'decimal_odds': best_decimal,
                    'ev': ev,
                    'kelly_bet': kelly,
                    'fanduel_odds': away_odds_available[
                        away_odds_available['bookmaker'] == 'fanduel'
                    ]['odds'].iloc[0] if 'fanduel' in away_odds_available['bookmaker'].values else None
                })
    
    result_df = pd.DataFrame(opportunities)
    
    # Calculate FanDuel advantage
    if 'fanduel_odds' in result_df.columns:
        result_df['fanduel_decimal'] = result_df['fanduel_odds'].apply(american_to_decimal)
        result_df['ev_fanduel'] = (result_df['model_prob'] * (result_df['fanduel_decimal'] - 1)) - ((1 - result_df['model_prob']) * 1)
        result_df['ev_gain_vs_fanduel'] = result_df['ev'] - result_df['ev_fanduel']
    
    return result_df.sort_values('ev', ascending=False)


if __name__ == "__main__":
    import os
    from src.utils.config import PREDICTIONS_CSV
    
    # Check for API key
    if not os.getenv('ODDS_API_KEY'):
        print("ERROR: ODDS_API_KEY environment variable not set")
        print("\nTo use this tool:")
        print("1. Sign up at https://the-odds-api.com/")
        print("2. Get your free API key (500 requests/month)")
        print("3. Set environment variable:")
        print("   export ODDS_API_KEY='your_key_here'")
        print("\nOr add to a .env file in the project root:")
        print("   ODDS_API_KEY=your_key_here")
        exit(1)
    
    # Load predictions
    if not PREDICTIONS_CSV.exists():
        print("No predictions file found. Generating predictions first...")
        predict_upcoming_games()
    
    predictions = pd.read_csv(PREDICTIONS_CSV)
    
    # Fetch live odds
    print("Fetching live odds from multiple sportsbooks...")
    odds_df = fetch_nfl_odds()
    
    print(f"\nFound odds from {len(odds_df['bookmaker'].unique())} bookmakers:")
    print(odds_df['bookmaker'].unique())
    
    # Find opportunities
    print("\n" + "="*100)
    print("BEST MONEYLINE OPPORTUNITIES (sorted by EV)")
    print("="*100)
    
    opportunities = find_best_moneyline_opportunities(predictions, odds_df)
    
    if opportunities.empty:
        print("No positive EV opportunities found")
    else:
        display_cols = ['team', 'opponent', 'location', 'model_prob', 'best_odds', 
                       'best_bookmaker', 'ev', 'kelly_bet']
        
        if 'fanduel_odds' in opportunities.columns:
            display_cols.extend(['fanduel_odds', 'ev_gain_vs_fanduel'])
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 150)
        print(opportunities[display_cols].to_string(index=False))
        
        print(f"\n\nTotal opportunities: {len(opportunities)}")
        print(f"Average EV: {opportunities['ev'].mean():.3f}")
        print(f"Total Kelly bet amount (on $100 bankroll): ${opportunities['kelly_bet'].sum():.2f}")
        
        if 'ev_gain_vs_fanduel' in opportunities.columns:
            better_books = opportunities[opportunities['ev_gain_vs_fanduel'] > 0]
            print(f"\nBets with better odds than FanDuel: {len(better_books)}/{len(opportunities)}")
            if not better_books.empty:
                print(f"Average EV gain by shopping odds: {better_books['ev_gain_vs_fanduel'].mean():.3f}")

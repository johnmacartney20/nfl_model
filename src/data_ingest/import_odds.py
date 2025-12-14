"""
Fetch odds from multiple sportsbooks using The Odds API.
Free tier: 500 requests/month
Sign up at: https://the-odds-api.com/
"""
import requests
import pandas as pd
from datetime import datetime
import os


def fetch_nfl_odds(api_key=None, regions='us', markets='h2h,spreads,totals'):
    """
    Fetch NFL odds from multiple sportsbooks.
    
    Args:
        api_key: The Odds API key (or set ODDS_API_KEY env var)
        regions: Geographic regions ('us', 'uk', 'au', or 'eu')
        markets: Bet types ('h2h' = moneyline, 'spreads', 'totals')
    
    Returns:
        DataFrame with odds from multiple sportsbooks
    """
    if api_key is None:
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("Must provide api_key or set ODDS_API_KEY environment variable")
    
    url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
    
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f'Failed to get odds: {response.status_code} - {response.text}')
    
    # Check remaining quota
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    print(f"API requests used: {used}, remaining: {remaining}")
    
    games = response.json()
    
    # Parse into structured format
    odds_data = []
    
    for game in games:
        game_id = game['id']
        commence_time = game['commence_time']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Process each bookmaker
        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker['key']
            
            # Extract odds for each market
            for market in bookmaker.get('markets', []):
                market_key = market['key']
                
                if market_key == 'h2h':  # Moneyline
                    for outcome in market['outcomes']:
                        team = outcome['name']
                        odds = outcome['price']
                        
                        odds_data.append({
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': book_name,
                            'market': 'moneyline',
                            'team': team,
                            'odds': odds
                        })
                
                elif market_key == 'spreads':  # Point spread
                    for outcome in market['outcomes']:
                        team = outcome['name']
                        spread = outcome['point']
                        odds = outcome['price']
                        
                        odds_data.append({
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': book_name,
                            'market': 'spread',
                            'team': team,
                            'spread': spread,
                            'odds': odds
                        })
                
                elif market_key == 'totals':  # Over/under
                    for outcome in market['outcomes']:
                        position = outcome['name']  # 'Over' or 'Under'
                        total = outcome['point']
                        odds = outcome['price']
                        
                        odds_data.append({
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': book_name,
                            'market': 'total',
                            'position': position,
                            'total': total,
                            'odds': odds
                        })
    
    return pd.DataFrame(odds_data)


def get_best_moneyline_odds(odds_df):
    """
    Find the best moneyline odds across all sportsbooks for each team.
    
    Args:
        odds_df: DataFrame from fetch_nfl_odds()
    
    Returns:
        DataFrame with best available odds per team
    """
    ml_odds = odds_df[odds_df['market'] == 'moneyline'].copy()
    
    # Group by game and team, find max odds (best payout)
    best_odds = ml_odds.loc[ml_odds.groupby(['game_id', 'team'])['odds'].idxmax()]
    
    return best_odds[['game_id', 'home_team', 'away_team', 'team', 'odds', 'bookmaker']]


def compare_to_fanduel(odds_df, fanduel_key='fanduel'):
    """
    Compare FanDuel odds to best available odds from other books.
    
    Args:
        odds_df: DataFrame from fetch_nfl_odds()
        fanduel_key: Bookmaker key for FanDuel (default 'fanduel')
    
    Returns:
        DataFrame showing FanDuel vs best alternative odds
    """
    ml_odds = odds_df[odds_df['market'] == 'moneyline'].copy()
    
    # Separate FanDuel and other books
    fanduel = ml_odds[ml_odds['bookmaker'] == fanduel_key].copy()
    others = ml_odds[ml_odds['bookmaker'] != fanduel_key].copy()
    
    # Get best odds from other books
    best_others = others.loc[others.groupby(['game_id', 'team'])['odds'].idxmax()]
    
    # Merge
    comparison = fanduel.merge(
        best_others[['game_id', 'team', 'odds', 'bookmaker']],
        on=['game_id', 'team'],
        suffixes=('_fanduel', '_best'),
        how='outer'
    )
    
    # Calculate advantage
    comparison['odds_advantage'] = comparison['odds_best'] - comparison['odds_fanduel']
    comparison['better_book_available'] = comparison['odds_advantage'] > 0
    
    return comparison.sort_values('odds_advantage', ascending=False)


if __name__ == "__main__":
    # Example usage
    try:
        print("Fetching NFL odds from multiple sportsbooks...")
        odds_df = fetch_nfl_odds()
        
        print(f"\nFetched {len(odds_df)} odds records")
        print(f"Bookmakers: {odds_df['bookmaker'].unique()}")
        
        # Show best moneyline odds
        print("\n" + "="*80)
        print("BEST MONEYLINE ODDS BY TEAM")
        print("="*80)
        best_ml = get_best_moneyline_odds(odds_df)
        print(best_ml.to_string(index=False))
        
        # Compare to FanDuel
        if 'fanduel' in odds_df['bookmaker'].values:
            print("\n" + "="*80)
            print("FANDUEL vs BEST AVAILABLE")
            print("="*80)
            comparison = compare_to_fanduel(odds_df)
            print(comparison[['home_team', 'away_team', 'team', 'odds_fanduel', 
                            'odds_best', 'bookmaker_best', 'odds_advantage']].head(20).to_string(index=False))
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo use this module:")
        print("1. Sign up at https://the-odds-api.com/")
        print("2. Get your API key")
        print("3. Set environment variable: export ODDS_API_KEY='your_key_here'")
        print("4. Or pass it directly: fetch_nfl_odds(api_key='your_key_here')")

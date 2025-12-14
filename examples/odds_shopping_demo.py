"""
Example: Using The Odds API to enhance your betting strategy

This demonstrates how line shopping can improve your expected value
even when your model already finds positive EV bets.
"""
from src.data_ingest.import_odds import fetch_nfl_odds
from src.modeling.find_best_odds import find_best_moneyline_opportunities
import pandas as pd
import os


def example_line_shopping_value():
    """
    Show how much value you can gain by shopping for the best odds.
    """
    
    # Simulate model predictions (normally from your model)
    sample_predictions = pd.DataFrame([
        {
            'home_team': 'BAL',
            'away_team': 'CIN', 
            'model_win_pct_home': 0.618,  # Model thinks BAL has 61.8% win prob
            'home_moneyline': -162,        # FanDuel has BAL at -162
        },
        {
            'home_team': 'BUF',
            'away_team': 'DET',
            'model_win_pct_home': 0.736,  # Model thinks BUF has 73.6% win prob
            'home_moneyline': 105,         # FanDuel has BUF at +105
        }
    ])
    
    print("="*80)
    print("EXAMPLE: Line Shopping Value")
    print("="*80)
    
    # Example 1: Favorite (Ravens -162 at FanDuel)
    print("\n📊 EXAMPLE 1: Baltimore Ravens")
    print("-" * 80)
    print(f"Model win probability: 61.8%")
    print(f"FanDuel odds: -162 (implied prob: 61.8%)")
    print(f"Break-even: Model needs to be right at exact market price")
    
    print("\nNow checking other sportsbooks...")
    print("DraftKings: -152 (implied prob: 60.3%)")
    print("Caesars: -155 (implied prob: 60.8%)")
    print("BetMGM: -158 (implied prob: 61.2%)")
    
    # Calculate EV difference
    fanduel_decimal = 1 + (100/162)  # 1.617
    draftkings_decimal = 1 + (100/152)  # 1.658
    
    fanduel_ev = (0.618 * (fanduel_decimal - 1)) - (0.382 * 1)
    draftkings_ev = (0.618 * (draftkings_decimal - 1)) - (0.382 * 1)
    
    print(f"\nExpected Value:")
    print(f"  FanDuel (-162):    {fanduel_ev:+.3f} ({fanduel_ev*100:+.1f}%)")
    print(f"  DraftKings (-152): {draftkings_ev:+.3f} ({draftkings_ev*100:+.1f}%)")
    print(f"  ⭐ Gain from shopping: {(draftkings_ev - fanduel_ev)*100:.1f}%")
    
    print(f"\nOn a $100 bet:")
    print(f"  FanDuel EV: ${fanduel_ev * 100:.2f}")
    print(f"  DraftKings EV: ${draftkings_ev * 100:.2f}")
    print(f"  💰 Extra profit by shopping: ${(draftkings_ev - fanduel_ev) * 100:.2f}")
    
    # Example 2: Underdog (Bills +105 at FanDuel)
    print("\n\n📊 EXAMPLE 2: Buffalo Bills")
    print("-" * 80)
    print(f"Model win probability: 73.6%")
    print(f"FanDuel odds: +105 (implied prob: 48.8%)")
    print(f"Model edge: 73.6% - 48.8% = 24.8% edge! 🎯")
    
    print("\nNow checking other sportsbooks...")
    print("Pinnacle: +110 (implied prob: 47.6%)")
    print("BetMGM: +108 (implied prob: 48.1%)")
    print("Caesars: +106 (implied prob: 48.5%)")
    
    fanduel_decimal = 1 + (105/100)  # 2.05
    pinnacle_decimal = 1 + (110/100)  # 2.10
    
    fanduel_ev = (0.736 * (fanduel_decimal - 1)) - (0.264 * 1)
    pinnacle_ev = (0.736 * (pinnacle_decimal - 1)) - (0.264 * 1)
    
    print(f"\nExpected Value:")
    print(f"  FanDuel (+105):  {fanduel_ev:+.3f} ({fanduel_ev*100:+.1f}%)")
    print(f"  Pinnacle (+110): {pinnacle_ev:+.3f} ({pinnacle_ev*100:+.1f}%)")
    print(f"  ⭐ Gain from shopping: {(pinnacle_ev - fanduel_ev)*100:.1f}%")
    
    print(f"\nOn a $100 bet:")
    print(f"  FanDuel EV: ${fanduel_ev * 100:.2f}")
    print(f"  Pinnacle EV: ${pinnacle_ev * 100:.2f}")
    print(f"  💰 Extra profit by shopping: ${(pinnacle_ev - fanduel_ev) * 100:.2f}")
    
    # Summary
    print("\n\n" + "="*80)
    print("💡 KEY TAKEAWAYS")
    print("="*80)
    print("1. Even small odds differences (5-10 points) add meaningful EV")
    print("2. Underdog bets benefit more from shopping (larger % gains)")
    print("3. Over a season, this compounds significantly:")
    print("   - 20 bets × $2-5 extra EV = $40-100 more profit")
    print("   - Cost: $0 (free API tier)")
    print("   - Time: 30 seconds per betting session")
    print("\n4. Pinnacle often has the best odds (lowest vig)")
    print("5. Compare at least 3-4 books for each bet")


def show_api_efficiency():
    """
    Show how efficiently The Odds API works with the free tier.
    """
    print("\n\n" + "="*80)
    print("📈 API USAGE EFFICIENCY")
    print("="*80)
    
    print("\nFree Tier: 500 requests/month")
    print("\nEach request returns:")
    print("  ✓ All upcoming NFL games (typically 16 games/week)")
    print("  ✓ All markets (moneyline, spread, totals)")
    print("  ✓ All 20+ sportsbooks")
    print("  ✓ Live, real-time odds")
    
    print("\nTypical usage pattern:")
    print("  • 1 request before making bets (Sunday morning)")
    print("  • 1 request for MNF/TNF if betting midweek")
    print("  = ~8 requests per month for weekly NFL betting")
    print("  = 492 requests remaining for other sports!")
    
    print("\nCost per bet:")
    print("  500 requests ÷ 52 weeks = 9.6 requests/week available")
    print("  If betting 5 games/week:")
    print("  500 requests ÷ 260 bets = 0.02 requests per bet")
    print("  = $0.00 per bet (free tier)")
    
    print("\nBottom line: The free tier is MORE than enough for NFL betting 🎉")


if __name__ == "__main__":
    # Show examples even without API key
    example_line_shopping_value()
    show_api_efficiency()
    
    # If API key is set, show real data
    if os.getenv('ODDS_API_KEY'):
        print("\n\n" + "="*80)
        print("📡 FETCHING REAL ODDS DATA")
        print("="*80)
        try:
            odds = fetch_nfl_odds(markets='h2h')  # Only moneylines for speed
            print(f"\n✓ Successfully fetched odds from {len(odds['bookmaker'].unique())} sportsbooks")
            print(f"✓ Covering {len(odds['game_id'].unique())} upcoming games")
            
            # Show bookmaker list
            print(f"\nAvailable sportsbooks:")
            for book in sorted(odds['bookmaker'].unique()):
                print(f"  • {book}")
                
        except Exception as e:
            print(f"\n❌ Error fetching odds: {e}")
    else:
        print("\n\n💡 To see real odds data:")
        print("1. Get free API key: https://the-odds-api.com/")
        print("2. Set environment variable: export ODDS_API_KEY='your_key'")
        print("3. Run this script again")

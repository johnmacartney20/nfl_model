# Multi-Sportsbook Odds Comparison

This module fetches real-time odds from multiple sportsbooks to find the best available prices for your model's value bets.

## Why This Matters

Line shopping can significantly improve your ROI:
- **Moneyline example**: FanDuel offers Chiefs -150, DraftKings offers -145
- Your model says bet Chiefs at 73% win probability
- FanDuel EV: 6.7%, DraftKings EV: 9.1% (2.4% improvement!)
- On a $100 bet, that's $2.40 more expected value

## Setup

### 1. Get API Key (Free)

Sign up at [The Odds API](https://the-odds-api.com/):
- Free tier: 500 requests/month
- Each request fetches all games and all books (very efficient)
- Covers 20+ major sportsbooks

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env and add your API key
```

Or set environment variable:

```bash
export ODDS_API_KEY='your_key_here'
```

## Usage

### Basic: Find Best Moneyline Odds

```bash
python3 -m src.modeling.find_best_odds
```

This will:
1. Load your model's predictions
2. Fetch current odds from all sportsbooks
3. Show the best available odds for each positive EV bet
4. Compare to FanDuel to show how much you gain by shopping

### Advanced: Integrate Into Pipeline

```python
from src.data_ingest.import_odds import fetch_nfl_odds
from src.modeling.find_best_odds import find_best_moneyline_opportunities
import pandas as pd

# Get your predictions
predictions = pd.read_csv('data/outputs/predictions_latest.csv')

# Fetch live odds
odds = fetch_nfl_odds()

# Find opportunities
opportunities = find_best_moneyline_opportunities(predictions, odds)

# Place bets at best books!
```

## Supported Sportsbooks

The Odds API includes 20+ books:
- **US**: FanDuel, DraftKings, BetMGM, Caesars, PointsBet, BetRivers, Unibet
- **International**: Bet365, William Hill, Ladbrokes, Pinnacle
- And more...

## Example Output

```
BEST MONEYLINE OPPORTUNITIES (sorted by EV)

team  opponent location model_prob best_odds best_bookmaker      ev  kelly_bet fanduel_odds ev_gain_vs_fanduel
 BAL       CIN     away      0.618      -152      draftkings   0.089      22.25         -162              0.012
  GB        TB     home      0.587      -135         betmgm   0.076      19.00         -140              0.008
 BUF       DET     home      0.736       110        pinnacle   0.071      17.75          105              0.006
```

**Translation**: 
- Bet Ravens ML at DraftKings (-152) instead of FanDuel (-162) → 1.2% more EV
- Your model gives Ravens 61.8% win probability
- Recommended bet: $22.25 on $100 bankroll (quarter-Kelly)

## API Usage Tips

- Each call to `fetch_nfl_odds()` counts as 1 request
- Fetch once per betting session (odds don't change that fast)
- 500 requests/month = ~16 per day, plenty for weekly betting
- Set `markets='h2h'` to only fetch moneylines (saves processing time)

## Cost Analysis

With your 91.7% moneyline win rate and 52% ROI:
- Average line shopping gain: ~1-3% EV improvement
- On $100/week betting: ~$100-300/year extra profit
- API cost: $0 (free tier)

**Return on effort**: Very high! 🎯

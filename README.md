# NFL Model Project

This repository contains code and data for building, training, and evaluating NFL game prediction models with multi-sportsbook odds comparison.

## Features

- **73.6% Win Prediction Accuracy** - Logistic regression model with EPA and DVOA features
- **52% ROI on Moneylines** - 91.7% win rate on positive EV bets
- **Multi-Sportsbook Comparison** - Find best odds across 20+ books
- **Kelly Criterion Sizing** - Risk-optimized bet sizing by bet type
- **EPA-Based Score Predictions** - Real predictive model (not Vegas line reversal)

## Structure

- `data/`: Raw, interim, feature, and output datasets
- `models/`: Trained models and evaluation reports
- `src/`: Source code for data ingestion, feature engineering, modeling, and pipeline
- `docs/`: Documentation for advanced features

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Predictions

```bash
# Run weekly pipeline (updates data, trains model, generates predictions)
python3 -m src.pipeline.run_weekly_pipeline

# Or just predict upcoming games
python3 -m src.modeling.predict_upcoming
```

### 3. Find Best Odds (Optional)

Line shopping can add 1-3% EV by finding better prices across sportsbooks:

```bash
# Setup (one-time)
cp .env.example .env
# Add your free API key from https://the-odds-api.com/

# Find best odds
./find_best_odds.sh
```

See [docs/ODDS_COMPARISON.md](docs/ODDS_COMPARISON.md) for details.

## Model Performance

**2025 Season (Weeks 1-14)**
- Win predictions: 73.6% accuracy
- Moneyline bets: 91.7% win rate, 52.1% ROI
- Spread bets: 75.0% win rate, 2.0% ROI
- Total bets: 69.2% win rate, 3.9% ROI

**Kelly Criterion Adjustments**
- Moneyline: 0.25 (strong edge)
- Spread: 0.15 (moderate edge)
- Total: 0.05 (minimal edge)

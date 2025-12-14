#!/bin/bash
# Quick script to find best odds for this week's value bets

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for API key
if [ -z "$ODDS_API_KEY" ]; then
    echo "❌ ODDS_API_KEY not found"
    echo ""
    echo "Setup instructions:"
    echo "1. Sign up at https://the-odds-api.com/"
    echo "2. Create .env file: cp .env.example .env"
    echo "3. Add your API key to .env"
    exit 1
fi

echo "🔍 Finding best odds across all sportsbooks..."
echo ""

python3 -m src.modeling.find_best_odds

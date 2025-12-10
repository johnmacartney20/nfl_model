# Model Evaluation Tools

## Evaluate Past Week Performance

Use `evaluate_past_week.py` to see what predictions the model would have made for completed games.

### Usage Examples

**1. Analyze most recent completed week:**
```bash
python3 -m src.modeling.evaluate_past_week
```

**2. Analyze a specific week:**
```bash
python3 -m src.modeling.evaluate_past_week 13
```

**3. Analyze a specific week from a different season:**
```bash
python3 -m src.modeling.evaluate_past_week 13 2024
```

**4. Analyze ALL completed weeks (full season performance):**
```bash
python3 -m src.modeling.evaluate_past_week all
```

### Output Includes

- Game-by-game predictions vs actual results
- Model score predictions vs book implied scores
- Win probability accuracy
- Over/Under prediction accuracy
- Cover (spread) prediction accuracy
- Best and worst predictions
- Week-by-week breakdown (when using "all" option)

### Example Output

```
WEEK 14 (2025) - RETROACTIVE PREDICTIONS vs ACTUAL RESULTS
===========================================================
home_team away_team  home_score  away_score  actual_total  model_home_score  model_away_score  ...

MODEL PERFORMANCE SUMMARY
=========================
Games Analyzed: 14

Win Predictions: 9/14 correct (64.3%)
Cover Predictions: 5/14 correct (35.7%)
Over/Under Predictions: 9/14 correct (64.3%)

Average Total Points Error:
  Model: 10.73 points
  Book:  9.14 points
```

### Notes

- The script only evaluates games that have already been completed (have final scores)
- Model predictions are based on team stats BEFORE each game was played
- Predictions are "retroactive" - showing what the model would have predicted before the games happened

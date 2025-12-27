"""Generate week-specific Kelly evaluation CSVs from predictions_latest.csv

Produces:
 - data/outputs/week{week}_kelly_detailed.csv
 - data/outputs/week{week}_kelly_placed.csv  (subset where stake > 0)
 - data/outputs/predictions_week{week}_reconstructed_totals.csv

Usage: python3 src/analysis/generate_week_kelly_evals.py --season 2025 --week 16
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--predictions-file', type=str, default=None, help='Optional predictions CSV to read instead of predictions_latest.csv')
    args = parser.parse_args()

    season = args.season
    week = args.week

    preds_path = OUT / 'predictions_latest.csv' if args.predictions_file is None else Path(args.predictions_file)
    if not preds_path.exists():
        print(f"Predictions file not found: {preds_path}")
        return

    df = pd.read_csv(preds_path)
    # filter for requested season/week
    df_sel = df[(df['season'] == season) & (df['week'] == week)].copy()
    if df_sel.empty:
        print(f"No predictions found for season={season} week={week}")
        return

    # local copy of calculate_kelly_fraction (keeps behavior consistent with predict_upcoming)
    def calculate_kelly_fraction(probability, decimal_odds, kelly_fraction=0.25, bet_type='moneyline'):
        import pandas as _pd
        if _pd.isna(probability) or _pd.isna(decimal_odds):
            return 0.0
        b = decimal_odds - 1
        p = probability
        q = 1 - p
        kelly = (b * p - q) / b
        if kelly <= 0:
            return 0.0
        if bet_type == 'moneyline':
            adjusted_fraction = kelly_fraction
        elif bet_type == 'spread':
            adjusted_fraction = kelly_fraction * 0.6
        else:
            adjusted_fraction = kelly_fraction * 0.2
        return kelly * adjusted_fraction

    rows = []
    for _, r in df_sel.iterrows():
        away = r['away_team']
        home = r['home_team']
        game_label = f"{away}@{home}"

        # moneyline
        p_home = r.get('model_win_pct_home', r.get('home_win_prob'))
        p_away = 1 - p_home if p_home is not None else r.get('model_win_pct_away')

        # helper to safe-get decimal odds
        def safe_dec(x):
            try:
                return float(x) if pd.notna(x) else None
            except Exception:
                return None

        entries = []
        # home ML
        entries.append({
            'game': game_label,
            'bet_type': 'home_ml',
            'prob': p_home,
            'dec': safe_dec(r.get('dec_home_moneyline') if 'dec_home_moneyline' in r else r.get('home_moneyline')),
            'ev_per_$': r.get('ev_home_ml'),
        })
        # away ML
        entries.append({
            'game': game_label,
            'bet_type': 'away_ml',
            'prob': p_away,
            'dec': safe_dec(r.get('dec_away_moneyline') if 'dec_away_moneyline' in r else r.get('away_moneyline')),
            'ev_per_$': r.get('ev_away_ml'),
        })

        # spreads
        entries.append({
            'game': game_label,
            'bet_type': 'home_spread',
            'prob': r.get('model_cover_pct_home'),
            'dec': safe_dec(r.get('dec_home_spread_odds') if 'dec_home_spread_odds' in r else r.get('home_spread_odds')),
            'ev_per_$': r.get('ev_home_spread'),
        })
        entries.append({
            'game': game_label,
            'bet_type': 'away_spread',
            'prob': 1 - r.get('model_cover_pct_home') if pd.notna(r.get('model_cover_pct_home')) else None,
            'dec': safe_dec(r.get('dec_away_spread_odds') if 'dec_away_spread_odds' in r else r.get('away_spread_odds')),
            'ev_per_$': r.get('ev_away_spread'),
        })

        # totals
        entries.append({
            'game': game_label,
            'bet_type': 'over',
            'prob': r.get('model_over_pct'),
            'dec': safe_dec(r.get('dec_over_odds') if 'dec_over_odds' in r else r.get('over_odds')),
            'ev_per_$': r.get('ev_over'),
        })
        entries.append({
            'game': game_label,
            'bet_type': 'under',
            'prob': r.get('model_under_pct'),
            'dec': safe_dec(r.get('dec_under_odds') if 'dec_under_odds' in r else r.get('under_odds')),
            'ev_per_$': r.get('ev_under'),
        })

        for e in entries:
            prob = e['prob']
            dec = e['dec']
            bet_type = e['bet_type']
            try:
                k_frac = float(calculate_kelly_fraction(prob, dec, bet_type=('moneyline' if 'ml' in bet_type else ('total' if bet_type in ['over','under'] else 'spread'))))
            except Exception:
                k_frac = 0.0

            stake = round(k_frac * 100.0, 2)

            rows.append({
                'game': e['game'],
                'bet_type': bet_type,
                'prob': e['prob'],
                'dec': e['dec'],
                'ev_per_$': e['ev_per_$'],
                'kelly_frac': k_frac,
                'stake_$': stake,
                'stake_reflected': stake,
                'profit': 0.0,
                'away_team': away,
                'home_team': home,
            })

    out_df = pd.DataFrame(rows)

    detailed_path = OUT / f'week{week}_kelly_detailed.csv'
    out_df.to_csv(detailed_path, index=False)
    print(f"Wrote detailed bets to {detailed_path}")

    placed_df = out_df[out_df['stake_$'] > 0].copy()
    placed_path = OUT / f'week{week}_kelly_placed.csv'
    placed_df.to_csv(placed_path, index=False)
    print(f"Wrote placed bets (recommended) to {placed_path} ({len(placed_df)} rows)")

    # Reconstructed totals summary
    totals = df_sel[['away_team', 'home_team', 'total_line']].copy()
    totals = totals.rename(columns={'total_line': 'total'})
    totals['model_total_points'] = df_sel['model_total_points']
    totals['over_prob'] = df_sel.get('model_over_pct')
    totals['dec_over'] = df_sel.get('dec_over_odds')
    totals['ev_over'] = df_sel.get('ev_over')
    totals['kelly_over_$'] = df_sel.get('kelly_over')
    # placeholders for actual outcomes (not available yet)
    totals['actual_total'] = pd.NA
    totals['actual_outcome'] = pd.NA
    totals['abs_err_model'] = pd.NA
    totals['abs_err_book'] = pd.NA
    totals['model_predicts_over'] = totals['over_prob'] > 0.5
    totals['model_over_correct'] = pd.NA

    totals_path = OUT / f'predictions_week{week}_reconstructed_totals.csv'
    totals.to_csv(totals_path, index=False)
    print(f"Wrote reconstructed totals to {totals_path}")


if __name__ == '__main__':
    main()

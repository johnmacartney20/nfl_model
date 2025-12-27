import math
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"
RAW = ROOT / "data" / "raw"


def load_latest_schedule_score(home, away, season=None, week=None):
    sched = pd.read_csv(RAW / "schedules_2015_2025.csv")
    # normalize strings
    sched['home_team'] = sched['home_team'].astype(str).str.strip()
    sched['away_team'] = sched['away_team'].astype(str).str.strip()
    matches = sched[(sched['home_team'] == home) & (sched['away_team'] == away) & sched['home_score'].notna() & sched['away_score'].notna()].copy()
    if matches.empty:
        return None
    # if season/week provided, prefer exact match
    if season is not None:
        try:
            matches['season'] = matches['season'].astype(int)
        except Exception:
            pass
        matches = matches[matches['season'] == int(season)]
    if week is not None:
        try:
            matches['week'] = matches['week'].astype(int)
        except Exception:
            pass
        matches = matches[matches['week'] == int(week)]
    if matches.empty:
        return None
    # pick the most recent season/week among remaining matches
    try:
        matches['season'] = matches['season'].astype(int)
        matches['week'] = matches['week'].astype(int)
        matches = matches.sort_values(['season', 'week'], ascending=[False, False])
    except Exception:
        pass
    r = matches.iloc[0]
    return int(r['home_score']), int(r['away_score'])


def main():
    parser = argparse.ArgumentParser(description='Recompute placed bet outcomes for a given week')
    parser.add_argument('--season', type=int, default=None, help='Season year (e.g. 2025)')
    parser.add_argument('--week', type=int, required=True, help='NFL week number (e.g. 16)')
    parser.add_argument('--placed-file', type=str, default=None, help='Optional path to placed bets CSV')
    args = parser.parse_args()

    season = args.season
    week = args.week

    placed_path = OUT / (f"week{week}_kelly_placed.csv" if args.placed_file is None else args.placed_file)
    if not placed_path.exists():
        print(f"Placed bets file not found: {placed_path}")
        return

    placed = pd.read_csv(placed_path)
    # Parse game into away@home
    placed[['away_team', 'home_team']] = placed['game'].str.split('@', expand=True)
    # load predictions to get lines
    preds_path = OUT / "predictions_latest.csv"
    preds = None
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        preds['home_team'] = preds['home_team'].astype(str).str.strip()
        preds['away_team'] = preds['away_team'].astype(str).str.strip()

    results = []
    for _, row in placed.iterrows():
        away = str(row['away_team']).strip()
        home = str(row['home_team']).strip()
        scores = load_latest_schedule_score(home, away, season=season, week=week)
        if scores is None:
            # no final score found
            profit = None
            outcome = 'no_score'
        else:
            home_score, away_score = scores
            margin = home_score - away_score
            total_pts = home_score + away_score

            bet = row['bet_type']
            stake = float(row.get('stake_$', row.get('stake', 0)))
            dec = row.get('dec')
            try:
                dec = float(dec)
            except Exception:
                dec = None

            # find relevant line from predictions if available
            spread_line = None
            total_line = None
            if preds is not None:
                match = preds[(preds['home_team'] == home) & (preds['away_team'] == away)]
                if not match.empty:
                    m = match.iloc[0]
                    spread_line = m.get('spread_line')
                    total_line = m.get('total_line')

            # default profit None
            profit = None

            if bet in ('home_ml', 'away_ml'):
                # resolve winner
                if home_score > away_score:
                    winner = 'home'
                elif away_score > home_score:
                    winner = 'away'
                else:
                    winner = 'push'

                bet_side = 'home' if bet == 'home_ml' else 'away'
                if winner == 'push':
                    profit = 0.0
                    outcome = 'push'
                elif winner == bet_side:
                    # win
                    if dec is None or dec <= 0:
                        profit = None
                        outcome = 'win'
                    else:
                        profit = stake * (dec - 1)
                        outcome = 'win'
                else:
                    profit = -stake
                    outcome = 'loss'

            elif bet in ('home_spread', 'away_spread'):
                if spread_line is None or pd.isna(spread_line):
                    profit = None
                    outcome = 'no_line'
                else:
                    # home covers if margin > spread_line
                    try:
                        spread_line = float(spread_line)
                    except Exception:
                        profit = None
                        outcome = 'bad_line'
                    else:
                        if margin == spread_line:
                            profit = 0.0
                            outcome = 'push'
                        else:
                            home_covers = (margin > spread_line)
                            bet_side = 'home' if bet == 'home_spread' else 'away'
                            side_won = (home_covers and bet_side == 'home') or (not home_covers and bet_side == 'away')
                            if side_won:
                                if dec is None or dec <= 0:
                                    profit = None
                                else:
                                    profit = stake * (dec - 1)
                                outcome = 'win'
                            else:
                                profit = -stake
                                outcome = 'loss'

            elif bet in ('over', 'under'):
                if total_line is None or pd.isna(total_line):
                    profit = None
                    outcome = 'no_line'
                else:
                    try:
                        total_line = float(total_line)
                    except Exception:
                        profit = None
                        outcome = 'bad_line'
                    else:
                        if total_pts == total_line:
                            profit = 0.0
                            outcome = 'push'
                        else:
                            over_won = total_pts > total_line
                            bet_side = 'over' if bet == 'over' else 'under'
                            if (over_won and bet_side == 'over') or (not over_won and bet_side == 'under'):
                                if dec is None or dec <= 0:
                                    profit = None
                                else:
                                    profit = stake * (dec - 1)
                                outcome = 'win'
                            else:
                                profit = -stake
                                outcome = 'loss'
            else:
                profit = None
                outcome = 'unknown_bet_type'

        results.append({**row.to_dict(), 'home_score': (scores[0] if scores is not None else None), 'away_score': (scores[1] if scores is not None else None), 'profit_recomputed': profit, 'outcome_recomputed': outcome})

    out_df = pd.DataFrame(results)
    # write corrected placed file
    corrected_path = OUT / f'week{week}_kelly_placed_corrected.csv'
    out_df.to_csv(corrected_path, index=False)
    print(f"Wrote corrected placed bets to {corrected_path}")

    # compute performance by category
    def category_of(bet):
        if bet in ('home_ml', 'away_ml'):
            return 'moneyline'
        if bet in ('home_spread', 'away_spread'):
            return 'spread'
        if bet in ('over', 'under'):
            return 'totals'
        return 'other'

    out_df['category'] = out_df['bet_type'].apply(category_of)
    # only include rows where profit_recomputed is numeric
    perf = out_df[out_df['profit_recomputed'].notna()].copy()
    perf['wagered'] = perf['stake_$'].astype(float)
    agg = perf.groupby('category').agg(n_bets=('game','count'), wagered=('wagered','sum'), profit=('profit_recomputed','sum'))
    agg['roi'] = agg['profit'] / agg['wagered']
    # wins / losses / pushes
    wins = perf[perf['profit_recomputed'] > 0].groupby('category').size().rename('wins')
    losses = perf[perf['profit_recomputed'] < 0].groupby('category').size().rename('losses')
    pushes = perf[perf['profit_recomputed'] == 0].groupby('category').size().rename('pushes')
    agg = agg.join(wins, how='left').join(losses, how='left').join(pushes, how='left').fillna(0)
    agg['winrate'] = agg['wins'] / (agg['wins'] + agg['losses']).replace({0: None})
    agg['avg_stake'] = agg['wagered'] / agg['n_bets']

    perf_path = OUT / f'week{week}_kelly_performance_by_type.csv'
    agg.reset_index().to_csv(perf_path, index=False)
    print(f"Wrote performance summary to {perf_path}")


if __name__ == '__main__':
    main()

"""Leakage, time-order, label and spread-sign tests for the game model."""
import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import build_team_game_epa as bte
from src.feature_engineering.build_team_game_epa import compute_pregame_team_features
from src.modeling.train_logistic_model import load_training_frame, walk_forward_splits, FEATURE_COLS
from src.utils.helpers import add_home_away_flags
from expected_points_model import add_book_implied_scores
from sim import simulate_game_outcomes


def _sched(rows):
    return pd.DataFrame(rows, columns=["season", "week", "game_id", "home_team", "away_team"])


def _tge(rows):
    """rows: (season, week, game_id, team, off_epa, def_epa, sr)"""
    return pd.DataFrame(rows, columns=["season", "week", "game_id", "team", "off_epa", "def_epa", "off_success_rate"])


# Two seasons. 2025 gives the prior; 2026 is the season under test.
SCHED = _sched([
    (2025, 1, "25_1", "AAA", "BBB"),
    (2025, 2, "25_2", "BBB", "AAA"),
    (2026, 1, "26_1", "AAA", "BBB"),
    (2026, 2, "26_2", "BBB", "AAA"),
    (2026, 3, "26_3", "AAA", "BBB"),
    (2026, 4, "26_4", "BBB", "AAA"),  # unplayed
])


def _tge_with(week3_aaa_off):
    return _tge([
        (2025, 1, "25_1", "AAA", 0.10, 0.00, 0.5), (2025, 1, "25_1", "BBB", 0.00, 0.10, 0.4),
        (2025, 2, "25_2", "AAA", 0.10, 0.00, 0.5), (2025, 2, "25_2", "BBB", 0.00, 0.10, 0.4),
        (2026, 1, "26_1", "AAA", 0.20, 0.00, 0.5), (2026, 1, "26_1", "BBB", 0.00, 0.20, 0.4),
        (2026, 2, "26_2", "AAA", 0.30, 0.00, 0.5), (2026, 2, "26_2", "BBB", 0.00, 0.30, 0.4),
        (2026, 3, "26_3", "AAA", week3_aaa_off, 0.00, 0.5), (2026, 3, "26_3", "BBB", 0.00, week3_aaa_off, 0.4),
    ])


def _aaa(feats, game_id):
    return feats[(feats.game_id == game_id) & (feats.team == "AAA")].iloc[0]


class PointInTimeFeatureTests(unittest.TestCase):
    def test_early_week_features_ignore_later_games(self):
        normal = compute_pregame_team_features(SCHED, _tge_with(0.10))
        blowout = compute_pregame_team_features(SCHED, _tge_with(5.00))
        for gid in ["26_1", "26_2", "26_3"]:
            for col in ["team_avg_off_epa", "off_dvoa", "def_dvoa"]:
                self.assertAlmostEqual(_aaa(normal, gid)[col], _aaa(blowout, gid)[col], msg=f"{gid} {col}")
        # ...but the later, unplayed week-4 game does see the week-3 result
        self.assertNotAlmostEqual(_aaa(normal, "26_4")["team_avg_off_epa"], _aaa(blowout, "26_4")["team_avg_off_epa"])

    def test_game_excludes_its_own_result(self):
        feats = compute_pregame_team_features(SCHED, _tge_with(5.00))
        row = _aaa(feats, "26_3")
        self.assertEqual(row["team_games_played_pre"], 2)
        k = bte.PRIOR_WEIGHT_GAMES
        prior = 0.05 + (1 - bte.PRIOR_REGRESSION) * (0.10 - 0.05)  # league mean 0.05 in 2025
        expected = (0.20 + 0.30 + k * prior) / (2 + k)
        self.assertAlmostEqual(row["team_avg_off_epa"], expected)

    def test_week_one_uses_only_prior_season(self):
        feats = compute_pregame_team_features(SCHED, _tge_with(0.10))
        row = _aaa(feats, "26_1")
        self.assertEqual(row["team_games_played_pre"], 0)
        prior = 0.05 + (1 - bte.PRIOR_REGRESSION) * (0.10 - 0.05)
        self.assertAlmostEqual(row["team_avg_off_epa"], prior)

    def test_first_season_week_one_has_no_features(self):
        feats = compute_pregame_team_features(SCHED, _tge_with(0.10))
        self.assertTrue(np.isnan(_aaa(feats, "25_1")["team_avg_off_epa"]))

    def test_no_season_wide_constant_features(self):
        """Regression guard for the original bug: identical features in every week."""
        feats = compute_pregame_team_features(SCHED, _tge_with(0.10))
        aaa26 = feats[(feats.team == "AAA") & (feats.season == 2026)]
        self.assertGreater(aaa26["team_avg_off_epa"].nunique(), 1)


class WalkForwardTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for season in [2021, 2022, 2023, 2024]:
            for week in range(1, 6):
                rows.append({"season": season, "week": week, "game_id": f"{season}_{week}"})
        df = pd.DataFrame(rows)
        # shuffle to prove the split does not rely on input order
        return df.sample(frac=1.0, random_state=0)

    def test_every_training_game_precedes_every_test_game(self):
        df = self._frame()
        folds = list(walk_forward_splits(df, min_train_seasons=2))
        self.assertEqual([f[0] for f in folds], [2023, 2024])
        for _, tr, te in folds:
            last_train = df.loc[tr, ["season", "week"]].apply(tuple, axis=1).max()
            first_test = df.loc[te, ["season", "week"]].apply(tuple, axis=1).min()
            self.assertLess(last_train, first_test)
            self.assertTrue(set(tr).isdisjoint(te))

    def test_folds_are_chronological(self):
        seasons = [s for s, _, _ in walk_forward_splits(self._frame(), min_train_seasons=1)]
        self.assertEqual(seasons, sorted(seasons))


class LabelTests(unittest.TestCase):
    def test_unplayed_games_have_missing_label(self):
        df = pd.DataFrame({
            "home_team": ["A", "B", "C"], "away_team": ["X", "Y", "Z"],
            "home_score": [24, 10, np.nan], "away_score": [17, 20, np.nan],
        })
        out = add_home_away_flags(df)
        self.assertEqual(out["home_win"].iloc[0], 1)
        self.assertEqual(out["home_win"].iloc[1], 0)
        self.assertTrue(pd.isna(out["home_win"].iloc[2]))

    def test_training_frame_drops_unplayed(self):
        base = {c: 0.0 for c in FEATURE_COLS}
        df = pd.DataFrame([
            {**base, "season": 2026, "week": 1, "game_id": "g1", "home_score": 21, "away_score": 14},
            {**base, "season": 2026, "week": 5, "game_id": "g5", "home_score": np.nan, "away_score": np.nan},
        ])
        df = add_home_away_flags(df)
        self.assertEqual(load_training_frame(df)["game_id"].tolist(), ["g1"])


class SpreadSignTests(unittest.TestCase):
    """nflverse spread_line > 0 means the home team is favored."""

    def test_book_implied_scores_favor_home_when_spread_positive(self):
        g = pd.DataFrame({"spread_line": [7.5], "total_line": [45.5]})
        out = add_book_implied_scores(g)
        self.assertAlmostEqual(out["book_home_score"].iloc[0] - out["book_away_score"].iloc[0], 7.5)

    def test_sim_cover_matches_predict_upcoming_rule(self):
        # Home expected to win by 10 with near-zero noise; line home -7.5 (spread_line=+7.5)
        g = pd.DataFrame({"model_home_score": [30.0], "model_away_score": [20.0],
                          "spread_line": [7.5], "total_line": [45.0]})
        out = simulate_game_outcomes(g, n_sims=2000, sigma=0.01)
        self.assertAlmostEqual(out["model_cover_pct_home"].iloc[0], 1.0)
        # Expected margin 10 does not cover a 12.5-point line
        g["spread_line"] = 12.5
        out = simulate_game_outcomes(g, n_sims=2000, sigma=0.01)
        self.assertAlmostEqual(out["model_cover_pct_home"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()

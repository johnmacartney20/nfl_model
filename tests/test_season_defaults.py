import datetime
import importlib
import sys
import types
import unittest
from unittest import mock

from src.utils.config import get_current_season


class GetCurrentSeasonTests(unittest.TestCase):
    def test_uses_prior_season_before_june(self):
        self.assertEqual(get_current_season(datetime.date(2026, 1, 15)), 2025)

    def test_uses_calendar_year_from_june_onward(self):
        self.assertEqual(get_current_season(datetime.date(2026, 9, 20)), 2026)


class EvaluatePastWeekDefaultSeasonTests(unittest.TestCase):
    def _import_module_with_stubs(self):
        pandas_stub = types.ModuleType("pandas")
        joblib_stub = types.ModuleType("joblib")
        ep_stub = types.ModuleType("expected_points_model")
        sim_stub = types.ModuleType("sim")

        ep_stub.add_book_implied_scores = lambda *args, **kwargs: None
        ep_stub.predict_scores_from_epa = lambda *args, **kwargs: None
        sim_stub.simulate_game_outcomes = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "pandas": pandas_stub,
                "joblib": joblib_stub,
                "expected_points_model": ep_stub,
                "sim": sim_stub,
            },
        ):
            sys.modules.pop("src.modeling.evaluate_past_week", None)
            return importlib.import_module("src.modeling.evaluate_past_week")

    def test_evaluate_past_week_uses_runtime_current_season_when_missing(self):
        module = self._import_module_with_stubs()

        with mock.patch.object(module, "get_current_season", side_effect=RuntimeError("season lookup")):
            with self.assertRaisesRegex(RuntimeError, "season lookup"):
                module.evaluate_past_week()

    def test_analyze_multiple_weeks_uses_runtime_current_season_when_missing(self):
        module = self._import_module_with_stubs()

        with mock.patch.object(module, "get_current_season", side_effect=RuntimeError("season lookup")):
            with self.assertRaisesRegex(RuntimeError, "season lookup"):
                module.analyze_multiple_weeks()


if __name__ == "__main__":
    unittest.main()

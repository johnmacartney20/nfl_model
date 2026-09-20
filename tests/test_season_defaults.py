import datetime
import importlib
import sys
import types
import unittest
from unittest import mock

import src.utils.config as config


class GetCurrentSeasonTests(unittest.TestCase):
    def test_uses_prior_season_before_june(self):
        self.assertEqual(config.get_current_season(datetime.date(2026, 1, 15)), 2025)

    def test_uses_calendar_year_from_june_onward(self):
        self.assertEqual(config.get_current_season(datetime.date(2026, 9, 20)), 2026)

    def test_excluded_weeks_follow_runtime_current_season(self):
        with mock.patch.object(config, "get_current_season", return_value=2026):
            excluded = config.get_excluded_reg_season_weeks_by_season()
        self.assertEqual(excluded[2026], [18])

    def test_excluded_weeks_constant_remains_dynamic(self):
        with mock.patch.object(config, "get_current_season", return_value=2026):
            self.assertEqual(config.EXCLUDE_REG_SEASON_WEEKS_BY_SEASON[2026], [18])


class EvaluatePastWeekDefaultSeasonTests(unittest.TestCase):
    class _FakeMask:
        def __and__(self, other):
            return self

    class _FakeSeries:
        def __init__(self, max_value=None):
            self.max_value = max_value

        def __eq__(self, other):
            return EvaluatePastWeekDefaultSeasonTests._FakeMask()

        def notna(self):
            return EvaluatePastWeekDefaultSeasonTests._FakeMask()

        def max(self):
            return self.max_value

    class _FakeCompletedFrame:
        empty = False

        def __getitem__(self, key):
            if key == "week":
                return EvaluatePastWeekDefaultSeasonTests._FakeSeries(max_value=3)
            raise KeyError(key)

    class _FakeScheduleFrame:
        def __getitem__(self, key):
            if isinstance(key, str):
                if key in {"season", "home_score"}:
                    return EvaluatePastWeekDefaultSeasonTests._FakeSeries()
                raise KeyError(key)
            return EvaluatePastWeekDefaultSeasonTests._FakeCompletedFrame()

    class _FakeWeekResult:
        empty = False

        def __setitem__(self, key, value):
            setattr(self, key, value)

    def _import_module_with_stubs(self):
        pandas_stub = types.ModuleType("pandas")
        joblib_stub = types.ModuleType("joblib")
        ep_stub = types.ModuleType("expected_points_model")
        sim_stub = types.ModuleType("sim")

        pandas_stub.read_csv = lambda *args, **kwargs: None
        pandas_stub.concat = lambda *args, **kwargs: None
        joblib_stub.load = lambda *args, **kwargs: None
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

    def test_analyze_multiple_weeks_passes_resolved_season_into_week_evaluations(self):
        module = self._import_module_with_stubs()
        calls = []

        def fake_eval(*, season, week):
            calls.append((season, week))
            return self._FakeWeekResult()

        with mock.patch.object(module, "get_current_season", return_value=2026), \
             mock.patch.object(module.pd, "read_csv", return_value=self._FakeScheduleFrame(), create=True), \
             mock.patch.object(module, "evaluate_past_week", side_effect=fake_eval), \
             mock.patch.object(module.pd, "concat", side_effect=RuntimeError("stop after loop"), create=True):
            with self.assertRaisesRegex(RuntimeError, "stop after loop"):
                module.analyze_multiple_weeks(start_week=1)

        self.assertEqual(calls, [(2026, 1), (2026, 2), (2026, 3)])


if __name__ == "__main__":
    unittest.main()

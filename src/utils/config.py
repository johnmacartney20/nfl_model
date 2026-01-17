from pathlib import Path
import os
import datetime

# Root directory is the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FEATURE_DIR = DATA_DIR / "features"
OUTPUT_DIR = DATA_DIR / "outputs"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
EVAL_DIR = MODELS_DIR / "eval"

def _infer_latest_season(today: datetime.date | None = None) -> int:
	"""Infer the most recent *season year* based on today's date.

	NFL seasons span calendar years (e.g., Jan 2026 is still the 2025 season).
	"""
	if today is None:
		today = datetime.date.today()
	# Rough heuristic: before June, we're still in the prior season's playoffs/offseason.
	return today.year - 1 if today.month < 6 else today.year


def _parse_seasons(value: str) -> list[int]:
	"""Parse seasons from an env string.

	Supports:
	- Comma-separated: "2021,2022,2023"
	- Ranges: "2021-2025"
	- Mixed: "2020-2022,2024,2025"
	"""
	seasons: set[int] = set()
	for part in (value or "").split(","):
		part = part.strip()
		if not part:
			continue
		if "-" in part:
			start_s, end_s = (p.strip() for p in part.split("-", 1))
			start = int(start_s)
			end = int(end_s)
			if start > end:
				start, end = end, start
			seasons.update(range(start, end + 1))
		else:
			seasons.add(int(part))
	return sorted(seasons)


def get_model_seasons() -> list[int]:
	"""Return the seasons used by the pipeline.

	Env overrides:
	- NFL_SEASONS="2021-2025" or "2021,2022,2023,2024,2025"
	- NFL_SEASON_START="2021" (uses inferred latest season)
	- NFL_SEASON_WINDOW="5" (uses inferred latest season)
	"""
	latest = _infer_latest_season()

	seasons_env = os.getenv("NFL_SEASONS")
	if seasons_env:
		parsed = _parse_seasons(seasons_env)
		if parsed:
			return parsed

	season_start_env = os.getenv("NFL_SEASON_START")
	if season_start_env:
		start = int(season_start_env)
		return list(range(start, latest + 1))

	window_env = os.getenv("NFL_SEASON_WINDOW")
	window = int(window_env) if window_env else 5
	start = max(1999, latest - window + 1)
	return list(range(start, latest + 1))


# Seasons you want in the model by default (configurable via env vars above)
DEFAULT_SEASONS = get_model_seasons()

# Some late-season weeks can be noisy (resting starters, etc.).
# These exclusions are applied in feature building so they do not:
# - influence season-to-date team EPA features
# - appear as training rows in game-level features
#
# Keys are NFL season year; values are a list of regular-season week numbers.
EXCLUDE_REG_SEASON_WEEKS_BY_SEASON = {
	2025: [18],
}

# File names
TEAM_GAME_EPA_CSV = INTERIM_DIR / "team_game_epa.csv"
GAME_LEVEL_FEATURES_CSV = FEATURE_DIR / "game_level_features.csv"
UPCOMING_GAMES_FEATURES_CSV = FEATURE_DIR / "upcoming_games_features.csv"

MODEL_PATH = TRAINED_MODELS_DIR / "home_win_logreg.pkl"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_latest.csv"

# QB adjustment feature flag and parameters
# If enabled, a league-wide backup effect (in points) will be applied to
# predicted team scores when the starting QB differs from the historical
# baseline for that team. The raw effect was estimated from pooled OLS
# and can be shrunk/capped here for conservatism.
ENABLE_QB_ADJUSTMENT = True
QB_BACKUP_EFFECT_RAW = -6.042  # raw pooled estimate (points lost when backup starts)
QB_SHRINK = 0.5  # shrink toward 0 to avoid over-adjusting
QB_CAP = 3.0     # cap the absolute adjustment (points)


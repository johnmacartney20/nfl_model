from pathlib import Path

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

# Seasons you want in the model for now
DEFAULT_SEASONS = list(range(2015, 2026))

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


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

# File names
TEAM_GAME_EPA_CSV = INTERIM_DIR / "team_game_epa.csv"
GAME_LEVEL_FEATURES_CSV = FEATURE_DIR / "game_level_features.csv"
UPCOMING_GAMES_FEATURES_CSV = FEATURE_DIR / "upcoming_games_features.csv"

MODEL_PATH = TRAINED_MODELS_DIR / "home_win_logreg.pkl"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_latest.csv"

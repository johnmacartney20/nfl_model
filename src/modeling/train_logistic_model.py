# trains baseline logistic regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import joblib

from src.utils.config import GAME_LEVEL_FEATURES_CSV, MODEL_PATH, TRAINED_MODELS_DIR
from src.utils.helpers import ensure_dirs

FEATURE_COLS = [
    "home_avg_off_epa",
    "home_avg_def_epa",
    "home_avg_success_rate",
    "away_avg_off_epa",
    "away_avg_def_epa",
    "away_avg_success_rate",
    "home_off_dvoa",
    "home_def_dvoa",
    "away_off_dvoa",
    "away_def_dvoa",
    "off_dvoa_diff",
    "def_dvoa_diff",
    "div_game",  # Division games play differently
    "home_rest",  # Rest advantage matters
    "away_rest",
]

TIME_KEYS = ["season", "week"]


def _make_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def walk_forward_splits(df: pd.DataFrame, min_train_seasons: int = 2):
    """Yield (test_season, train_idx, test_idx) in chronological order.

    Each fold trains on every game from seasons strictly before the test season
    and evaluates on the test season. No test game ever precedes a training game.
    """
    seasons = sorted(df["season"].unique())
    for test_season in seasons[min_train_seasons:]:
        train_idx = df.index[df["season"] < test_season]
        test_idx = df.index[df["season"] == test_season]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield test_season, train_idx, test_idx


def _metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, (p > 0.5).astype(int)),
        "max_prob": float(np.max(np.maximum(p, 1 - p))),
        "share_over_90": float(np.mean(np.maximum(p, 1 - p) > 0.90)),
    }


def load_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Only completed games with a label and full features
    df_train = df.dropna(subset=["home_score", "away_score", "home_win"])
    df_train = df_train.dropna(subset=FEATURE_COLS)
    sort_cols = TIME_KEYS + (["gameday"] if "gameday" in df_train.columns else []) + ["game_id"]
    return df_train.sort_values(sort_cols).reset_index(drop=True)


def walk_forward_evaluate(df_train: pd.DataFrame, min_train_seasons: int = 2) -> pd.DataFrame:
    rows = []
    for test_season, tr, te in walk_forward_splits(df_train, min_train_seasons):
        pipe = _make_pipeline()
        pipe.fit(df_train.loc[tr, FEATURE_COLS], df_train.loc[tr, "home_win"].astype(int))
        p = pipe.predict_proba(df_train.loc[te, FEATURE_COLS])[:, 1]
        y = df_train.loc[te, "home_win"].astype(int).to_numpy()
        m = _metrics(y, p)
        m["base_rate_log_loss"] = log_loss(y, np.full(len(y), df_train.loc[tr, "home_win"].astype(int).mean()), labels=[0, 1])
        m["test_season"] = int(test_season)
        rows.append(m)
    return pd.DataFrame(rows)


def train_logreg_model(min_train_seasons: int = 2):
    ensure_dirs([TRAINED_MODELS_DIR])
    df = pd.read_csv(GAME_LEVEL_FEATURES_CSV)
    df_train = load_training_frame(df)

    results = walk_forward_evaluate(df_train, min_train_seasons)
    if not results.empty:
        print("Walk-forward validation (train on prior seasons, test on next season):")
        cols = ["test_season", "n", "log_loss", "base_rate_log_loss", "brier", "accuracy", "max_prob", "share_over_90"]
        print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        w = results["n"] / results["n"].sum()
        print(
            f"Pooled: log_loss={np.average(results['log_loss'], weights=w):.3f} "
            f"brier={np.average(results['brier'], weights=w):.3f} "
            f"accuracy={np.average(results['accuracy'], weights=w):.3f}"
        )

    # Final model: fit on all completed games (all strictly before the slate being predicted)
    pipe = _make_pipeline()
    pipe.fit(df_train[FEATURE_COLS], df_train["home_win"].astype(int))
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return pipe, results

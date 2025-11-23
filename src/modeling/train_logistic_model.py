# trains baseline logistic regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

from src.utils.config import GAME_LEVEL_FEATURES_CSV, MODEL_PATH, TRAINED_MODELS_DIR
from src.utils.helpers import ensure_dirs

def train_logreg_model():
    ensure_dirs([TRAINED_MODELS_DIR])
    df = pd.read_csv(GAME_LEVEL_FEATURES_CSV)

    # Drop rows with missing scores (future games)
    df_train = df.dropna(subset=["home_score", "away_score"])

    feature_cols = [
    "home_avg_off_epa",
    "home_avg_def_epa",
    "home_avg_success_rate",
    "away_avg_off_epa",
    "away_avg_def_epa",
    "away_avg_success_rate",
    ]

    # Basic guard if some early seasons have missing EPA
    df_train = df_train.dropna(subset=feature_cols)

    X = df_train[feature_cols]
    y = df_train["home_win"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    val_acc = pipe.score(X_val, y_val)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Val accuracy:   {val_acc:.3f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

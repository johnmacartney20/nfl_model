# automated weekly ETL + prediction script
from src.pipeline.build_dataset import build_full_dataset
from src.modeling.train_logistic_model import train_logreg_model
from src.modeling.predict_upcoming import predict_upcoming_games

def run_weekly_pipeline():
    # You can tweak seasons here if you only want recent years
    build_full_dataset()
    train_logreg_model()
    predict_upcoming_games()
    print("Weekly pipeline complete.")

if __name__ == "__main__":
    run_weekly_pipeline()

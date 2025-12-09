# orchestrates full data build
from src.utils.config import DEFAULT_SEASONS
from src.data_ingest.import_pbp import import_pbp_data
from src.data_ingest.import_schedules import import_schedules
from src.feature_engineering.build_team_game_epa import build_team_game_epa
from src.feature_engineering.merge_features import build_game_level_features


def build_full_dataset(seasons=None):
    if seasons is None:
        seasons = DEFAULT_SEASONS

    pbp = import_pbp_data(seasons)
    sched = import_schedules(seasons)

    build_team_game_epa(pbp=pbp, seasons=seasons)

    # DVOA is now calculated internally from EPA data
    build_game_level_features(seasons)

    print("Full dataset build complete.")

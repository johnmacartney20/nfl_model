# helper functions (team mapping, cleaning)
import pandas as pd

def ensure_dirs(paths):
	for p in paths:
		p.mkdir(parents=True, exist_ok=True)

def add_home_away_flags(games_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Assumes games_df has:
	- home_team
	- away_team
	- home_score
	- away_score
	"""
	games_df = games_df.copy()
	games_df["home_win"] = (games_df["home_score"] > games_df["away_score"]).astype(int)
	return games_df

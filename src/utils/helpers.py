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

	Unplayed games (either score missing) get home_win = NA, not 0, so they
	can never be mistaken for home losses.
	"""
	games_df = games_df.copy()
	played = games_df["home_score"].notna() & games_df["away_score"].notna()
	home_win = (games_df["home_score"] > games_df["away_score"]).astype("Int64")
	games_df["home_win"] = home_win.where(played, pd.NA)
	return games_df

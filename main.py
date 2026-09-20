import nfl_data_py as nfl
import sys

from src.utils.config import CURRENT_SEASON


season = int(sys.argv[1]) if len(sys.argv) > 1 else CURRENT_SEASON
df = nfl.import_schedules([season])
print(df.head())

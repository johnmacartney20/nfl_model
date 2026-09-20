import nfl_data_py as nfl
import sys

from src.utils.config import get_current_season


season = int(sys.argv[1]) if len(sys.argv) > 1 else get_current_season()
df = nfl.import_schedules([season])
print(df.head())

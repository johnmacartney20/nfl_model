import nfl_data_py as nfl

df = nfl.import_schedules([2023])
print(df.head())

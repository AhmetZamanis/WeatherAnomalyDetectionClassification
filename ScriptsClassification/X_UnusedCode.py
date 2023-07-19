

# Retrieve features for each subsequence: Dataframe with each row as one sequence, 
# each column one feature, and each cell a pd.Series of N length
def get_series(g, name):
  return pd.Series(data = g[name].values, name = name)

x = df.groupby(["LOCATION", "ROWGROUP"], as_index = False).apply(lambda g: pd.Series(dict(
  mean_temp = get_series(g, "MEAN_TEMPERATURE"),
  total_precip = get_series(g, "TOTAL_PRECIPITATION"),
  month_sin = get_series(g, "month_sin"),
  month_cos = get_series(g, "month_cos"),
  week_sin = get_series(g, "week_sin"),
  week_cos = get_series(g, "week_cos"),
  day_sin = get_series(g, "day_sin"),
  day_cos = get_series(g, "day_cos"),
    )
  )
).drop(["LOCATION", "ROWGROUP"], axis = 1)

# Check datatype & index of cells
type(x.iloc[1,0])

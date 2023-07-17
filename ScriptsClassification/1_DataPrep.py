# Canadian weather data - Data prep for TS classification
# Data source: https://openml.org/search?type=data&status=active&id=43843


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import arff


# Set print options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)


# Set plotting options
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.autolayout"] = True
sns.set_style("darkgrid")


# Load raw data
raw_data = arff.load(open("./InputData/canadian_climate.arff", "r"))
df = pd.DataFrame(raw_data["data"], columns = [x[0] for x in raw_data["attributes"]])


# Check missing values: Pre-1960 locations: Calgary, Vancouver, Ottawa, Toronto
pd.isnull(df).sum()


# Wide to long conversion
df = pd.wide_to_long(
  df, stubnames = ["MEAN_TEMPERATURE", "TOTAL_PRECIPITATION"],
  i = "LOCAL_DATE", j = "LOCATION", sep = "_", suffix = r"\w+")
df = df.reset_index()


# Select observations only for Ottawa, Toronto, Vancouver
df = df[df["LOCATION"].isin(["OTTAWA", "TORONTO", "VANCOUVER"])]


# Convert LOCAL_DATE to datetime, set index for NA interpolation
df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])
df = df.set_index("LOCAL_DATE")


# Interpolate missing values in Ottawa, Toronto, Vancouver
df = df.groupby("LOCATION", group_keys = False).apply(
  lambda g: g.interpolate(method = "time"))
df = df.reset_index()


# Add cyclic terms for month, week of year and day of year
df["month_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["week_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.week / 53)
df["week_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.week / 53)
df["day_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)
df["day_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)


# Convert to N-day sequences in sktime format
n = 28

# Enumerate days for each city
df["DAYCOUNT"] = df.groupby("LOCATION").LOCATION.cumcount().add(1)

# Get rowgroups for each day
df["ROWGROUP"] = (df["DAYCOUNT"] // (n + 1)).astype(str)
df = df.drop("DAYCOUNT", axis = 1)

# Eliminate rowgroups which are not of length N
len(df[df["ROWGROUP"] == "0"]) / 3 # Length N
len(df[df["ROWGROUP"] == "1"]) / 3 # Length N+1
len(df[df["ROWGROUP"] == "1006"]) / 3 # Length N+1
len(df[df["ROWGROUP"] == "1007"]) / 3 # Length <N
df = df.loc[~df["ROWGROUP"].isin(["0", "1007"])]

# Retrieve targets for each subsequence
y = df.groupby(["LOCATION", "ROWGROUP"]).head(1)["LOCATION"]
y = y.reset_index().drop("index", axis = 1)

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

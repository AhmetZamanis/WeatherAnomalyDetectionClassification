# Canadian weather data - Data prep for TS classification
# Data source: https://openml.org/search?type=data&status=active&id=43843


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import arff

from X_HelperFunctionsClassif import plot_confusion
from sklearn.metrics import accuracy_score, log_loss
# from sklearn.model_selection import train_test_split


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
n = 27

# Enumerate days for each city
df["DAYCOUNT"] = df.groupby("LOCATION").LOCATION.cumcount().add(1)

# Get rowgroups for each day
df["ROWGROUP"] = (df["DAYCOUNT"] // (n + 1)).astype(str)
df = df.drop("DAYCOUNT", axis = 1)

# Eliminate rowgroups which are not of length N
len(df[df["ROWGROUP"] == "0"]) / 3 # Length N
len(df[df["ROWGROUP"] == "1"]) / 3 # Length N+1
len(df[df["ROWGROUP"] == "1042"]) / 3 # Length N+1
len(df[df["ROWGROUP"] == "1043"]) / 3 # Length <N
df = df.loc[~df["ROWGROUP"].isin(["0", "1043"])]

# Retrieve targets for each subsequence
y = df.groupby(["LOCATION", "ROWGROUP"]).head(1)["LOCATION"]
y = y.reset_index().drop("index", axis = 1).values.flatten()


# Retrieve features as 3Darray of shape (n_sequences, n_dimensions, seq_length)

# 2D arrays of (n_dimensions, seq_length) for each sequence
x = df.groupby(["LOCATION", "ROWGROUP"], as_index = False).apply(lambda g: np.array(
  [g["MEAN_TEMPERATURE"].values,
  g["TOTAL_PRECIPITATION"].values,
  g["month_sin"].values,
  g["month_cos"].values,
  g["week_sin"].values,
  g["week_cos"].values,
  g["day_sin"].values,
  g["day_cos"].values
    ]
  )
)

# 3Darray
x = np.array([x[i] for i in range(0, len(x))])


# Split train & test (most recent 20% sequences for all cities as test)

# Get indices
l = len(y)
len_test = int(l / 3 * 0.2)
len_train = int(l / 3 - len_test)
j = int(l / 3)
idx_train = list(range(0, len_train)) + list(range(j, len_train + j)) + list(range(j * 2, len_train + (j * 2)))
idx_test = list(range(0, l))
idx_test = list(set(idx_test).difference(idx_train))

# Perform split
y_train, y_test = y[idx_train], y[idx_test]
x_train, x_test = x[idx_train], x[idx_test]


# Get class labels
classes = np.unique(y_train)

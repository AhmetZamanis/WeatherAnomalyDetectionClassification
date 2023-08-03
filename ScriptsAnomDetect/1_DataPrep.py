# Canadian weather data - Data prep for TS anomaly detection
# Data source: https://openml.org/search?type=data&status=active&id=43843


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import arff
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller


# Set print options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)


# Set plotting options
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.autolayout"] = True
sns.set_style("darkgrid")
px_width = 800
px_height = 800


# Load raw data (package liac-arff)
raw_data = arff.load(open("./InputData/canadian_climate.arff", "r"))
df = pd.DataFrame(raw_data["data"], columns = [x[0] for x in raw_data["attributes"]])


# Convert LOCAL_DATE to datetime
df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])


# Add cyclic terms for month, week of year and day of year
df["month_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["week_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.isocalendar().week / 53)
df["week_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.isocalendar().week / 53)
df["day_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)
df["day_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)


# Retrieve darts TS for Edmonton (1960 start) and Ottawa (1940 start)
ts_edmonton = TimeSeries.from_dataframe(
  df,
  time_col = "LOCAL_DATE",
  value_cols = ["MEAN_TEMPERATURE_EDMONTON", "TOTAL_PRECIPITATION_EDMONTON"],
  fill_missing_dates = True
)
ts_edmonton = ts_edmonton.drop_before(pd.Timestamp("1960-12-31"))

ts_ottawa = TimeSeries.from_dataframe(
  df,
  time_col = "LOCAL_DATE",
  value_cols = ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"],
  fill_missing_dates = True
)


# Retrieve date covariates
ts_covars = TimeSeries.from_dataframe(
  df,
  time_col = "LOCAL_DATE",
  value_cols = ['month_sin', 'month_cos', 'week_sin', 'week_cos', 'day_sin', 
  'day_cos'],
  fill_missing_dates = True
)


# Interpolate all missing values for both series
na_filler = MissingValuesFiller()
ts_edmonton = na_filler.transform(ts_edmonton)
ts_ottawa = na_filler.transform(ts_ottawa)


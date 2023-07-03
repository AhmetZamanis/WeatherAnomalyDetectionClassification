# Canadian weather data - Dynamic time warping
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from darts.dataprocessing import dtw
import xarray as xr


# Retrieve first 2 weeks of December 2019, December 2018, Ottawa
series1 = ts_ottawa.drop_before(pd.Timestamp("2019-11-30")).drop_after(pd.Timestamp("2019-12-15"))
series2 = ts_ottawa.drop_before(pd.Timestamp("2018-11-30")).drop_after(pd.Timestamp("2019-01-01"))

# # Convert to range index
# series1 = TimeSeries.from_values(series1.values()).with_columns_renamed(
#   ["0", "1"], ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])
# series2 = TimeSeries.from_values(series2.values()).with_columns_renamed(
#   ["0", "1"], ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])


# Perform DTW
alignment = dtw.dtw(
  series1, 
  series2
  # window = dtw.Itakura(max_slope = 2) # 1.5 is fast & narrow, 3 is slow & wide
  )


# Get the alignment path: Indices of series 1 points and the indices of aligned 
# series 2 points
path = alignment.path()


# Visualize the alignment (works with custom fix)
alignment.plot_alignment(series2_y_offset = 100)
_ = plt.gca().set_title("Alignment")
plt.show()
plt.close("all")








# Breakdown of method: Works for DateTimeIndex, RangeIndex
series1_y_offset = 0
series2_y_offset = 100

(component1, component2) = ["MEAN_TEMPERATURE_OTTAWA", "MEAN_TEMPERATURE_OTTAWA"]

if not series1.is_univariate:
    series1 = series1.univariate_component(component1)
if not series2.is_univariate:
    series2 = series2.univariate_component(component2)

series1 += series1_y_offset
series2 += series2_y_offset

xa1 = series1.data_array(copy=False)
xa2 = series2.data_array(copy=False)

path = alignment.path()
n = len(path)

time_dim1 = series1._time_dim
time_dim2 = series2._time_dim

x_coords1 = np.array(xa1[time_dim1], dtype=xa1[time_dim1].dtype)[path[:, 0]]
x_coords2 = np.array(xa2[time_dim2], dtype=xa2[time_dim2].dtype)[path[:, 1]]

y_coords1 = series1.univariate_values()[path[:, 0]]
y_coords2 = series2.univariate_values()[path[:, 1]]

x_coords = np.zeros(n * 3, dtype=xa1[time_dim1].dtype)
y_coords = np.zeros(n * 3, dtype=np.float64)

x_coords[0::3] = x_coords1
x_coords[1::3] = x_coords2
if x_coords.dtype == "datetime64[s]": # With np.zeros, we don't need this step for RangeIndex series
  x_coords[2::3] = np.datetime64("NaT")

y_coords[0::3] = y_coords1
y_coords[1::3] = y_coords2
y_coords[2::3] = np.nan

arr = xr.DataArray(y_coords, dims=["value"], coords={"value": x_coords})
xr.plot.line(arr, x="value")

series1.plot()
series2.plot()
plt.show()
plt.close("all")






# UNUSED CODE
x_coords[2::3] = np.nan

if x_coords.dtype == "int64":
  x_coords[2::3] = np.nan
  
if x_coords.dtype == "datetime64[s]":
  x_coords[2::3] = np.datetime64("NaT")

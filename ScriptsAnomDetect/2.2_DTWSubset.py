# Canadian weather data - Dynamic time warping
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from darts.dataprocessing import dtw


# Retrieve first 2 weeks of December 2019, December 2018, Ottawa
ts_ottawa19 = ts_ottawa.drop_before(pd.Timestamp("2019-11-30")).drop_after(pd.Timestamp("2019-12-15"))
ts_ottawa18 = ts_ottawa.drop_before(pd.Timestamp("2018-11-30")).drop_after(pd.Timestamp("2019-01-01"))

ts_ottawa19 = TimeSeries.from_values(ts_ottawa19.values()).with_columns_renamed(
  ["0", "1"], ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])
ts_ottawa18 = TimeSeries.from_values(ts_ottawa18.values()).with_columns_renamed(
  ["0", "1"], ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])


# Plot raw series
fig, ax = plt.subplots(2, sharex = True)

ts_ottawa18["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2018", ax = ax[0])
ts_ottawa19["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2019", ax = ax[0])
_ = ax[0].set_title("Mean temperature, Ottawa")
_ = ax[0].set_xlabel("Day of december")

ts_ottawa18["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2018", ax = ax[1])
ts_ottawa19["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2019", ax = ax[1])
_ = ax[1].set_title("Total precipitation, Ottawa")
_ = ax[1].set_xlabel("Day of december")

plt.show()
plt.close("all")


# Perform DTW, plot cost matrix
alignment = dtw.dtw(
  ts_ottawa19, 
  ts_ottawa18
  # window = dtw.Itakura(max_slope = 2) # 1.5 is fast & narrow, 3 is slow & wide
  )
alignment.plot()
_ = plt.xlabel("Days of December 2019")
_ = plt.ylabel("Days of December 2018")
_ = plt.title("Cost matrix. Darker = Lower cost, higher similarity")
plt.show()
plt.close("all")


# Get DTW distance metrics
alignment.distance() # Total distances between each pair
alignment.mean_distance() # Mean distance between pairs


# Get the alignment path: Indices of series 1 points and the indices of aligned 
# series 2 points
path = alignment.path()


# Plot the alignments
alignment.plot_alignment(
  series2_y_offset = 100, components = ["MEAN_TEMPERATURE_OTTAWA", "MEAN_TEMPERATURE_OTTAWA"])
_ = plt.gca().set_title("Alignment, mean temperature, series 2 offset = 50")
plt.show()
plt.close("all")

alignment.plot_alignment(
  series2_y_offset = 100, components = ["TOTAL_PRECIPITATION_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])
_ = plt.gca().set_title("Alignment, total precipitation, series 2 offset = 100")
plt.show()
plt.close("all")


# Produce & plot warped series
ts_warped_19, ts_warped_18 = alignment.warped()
fig, ax = plt.subplots(2, sharex = True)

ts_warped_18["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2018", ax = ax[0])
ts_warped_19["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2019", ax = ax[0])
_ = ax[0].set_title("Mean temperature warped, Ottawa")
_ = ax[0].set_xlabel("Day of december")

ts_warped_18["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2018", ax = ax[1])
ts_warped_19["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2019", ax = ax[1])
_ = ax[1].set_title("Total precipitation warped, ottawa")
_ = ax[1].set_xlabel("Day of december")

plt.show()
plt.close("all")

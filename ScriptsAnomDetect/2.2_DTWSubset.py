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


# Get dataframes where rows = "ottawa19", "ottawa18", cols = Time, Temp, Precip (+25 for ottawa18)
y_offset = 100
df_alignment = []
for pair in path:
  idx1 = int(pair[0])
  idx2 = int(pair[1])
  # t1 = idx1 + 1 
  # t2 = idx2 + 1
  temp1 = ts_ottawa19["MEAN_TEMPERATURE_OTTAWA"][idx1].univariate_values()[0]
  temp2 = ts_ottawa18["MEAN_TEMPERATURE_OTTAWA"][idx2].univariate_values()[0] + y_offset
  precip1 = ts_ottawa19["TOTAL_PRECIPITATION_OTTAWA"][idx1].univariate_values()[0]
  precip2 = ts_ottawa18["TOTAL_PRECIPITATION_OTTAWA"][idx2].univariate_values()[0] + y_offset
  
  data = pd.DataFrame({
    "Time": [idx1, idx2],
    "MeanTemp": [temp1, temp2],
    "TotalPrecip": [precip1, precip2]
    }
  )
  df_alignment.append(data)


# Plot the alignments
fig, ax = plt.subplots(2)

# Mean temp
(ts_ottawa18["MEAN_TEMPERATURE_OTTAWA"] + y_offset).plot(
  label = "2018", ax = ax[0], marker = "o")
ts_ottawa19["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2019", ax = ax[0], marker = "o")
for pair in df_alignment:
  _ = ax[0].plot("Time", "MeanTemp", data = pair, c = "black", lw = 0.75)

_ = ax[0].set_title("Alignment plot, mean temperature Ottawa")
_ = ax[0].set_ylabel("Mean temp. (+100 for 2018)")
_ = ax[0].set_xlabel("Day of december")

# Total precip
(ts_ottawa18["TOTAL_PRECIPITATION_OTTAWA"] + y_offset).plot(
  label = "2018", ax = ax[1], marker = "o")
ts_ottawa19["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2019", ax = ax[1], marker = "o")
for pair in df_alignment:
  _ = ax[1].plot("Time", "TotalPrecip", data = pair, c = "black", lw = 0.75)

_ = ax[1].set_title("Alignment plot, total precipitation Ottawa")
_ = ax[1].set_ylabel("Total precip. (+100 for 2018)")
_ = ax[1].set_xlabel("Day of december")

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

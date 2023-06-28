# Canadian weather data - Dynamic time warping
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from darts.dataprocessing import dtw


# Plot raw Edmonton & Ottawa series
fig, ax = plt.subplots(2, sharex = True)

ts_ottawa["MEAN_TEMPERATURE_OTTAWA"].plot(label = "Ottawa", ax = ax[0])
ts_edmonton["MEAN_TEMPERATURE_EDMONTON"].plot(label = "Edmonton", ax = ax[0])
_ = ax[0].set_title("Mean temperature")

ts_ottawa["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "Ottawa", ax = ax[1])
ts_edmonton["TOTAL_PRECIPITATION_EDMONTON"].plot(label = "Edmonton", ax = ax[1])
_ = ax[1].set_title("Total precipitation")

plt.show()
plt.close("all")


# Perform DTW, plot cost matrix
alignment = dtw.dtw(
  ts_edmonton, 
  ts_ottawa,
  window = dtw.Itakura(max_slope = 2) # 1.5 is fast & narrow, 3 is slow & wide
  )
alignment.plot()
_ = plt.xlabel("Edmonton")
_ = plt.ylabel("Ottawa")
plt.show()
plt.close("all")


# Get DTW distance metrics
alignment.distance() # Total distances between each pair
alignment.mean_distance() # Mean distance between pairs


# Get the alignment path: Indices of series 1 points and the indices of aligned 
# series 2 points
path = alignment.path()


# Visualize the alignment
alignment.plot_alignment(
  series2_y_offset = 50, components = ["MEAN_TEMPERATURE_EDMONTON", "MEAN_TEMPERATURE_OTTAWA"])
_ = plt.gca().set_title("Alignment, mean temperature, series 2 offset = 50")
plt.show()
plt.close("all")

alignment.plot_alignment(
  series2_y_offset = 50, components = ["TOTAL_PRECIPITATION_EDMONTON", "TOTAL_PRECIPITATION_OTTAWA"])
_ = plt.gca().set_title("Alignment, total precipitation, series 2 offset = 100")
plt.show()
plt.close("all")


# Produce & plot warped series
ts_warped_ed, ts_warped_ottawa = alignment.warped()
fig, ax = plt.subplots(2, sharex = True)

ts_warped_ottawa["MEAN_TEMPERATURE_OTTAWA"].plot(label = "Ottawa", ax = ax[0])
ts_warped_ed["MEAN_TEMPERATURE_EDMONTON"].plot(label = "Edmonton", ax = ax[0])
_ = ax[0].set_title("Mean temperature, warped")

ts_warped_ottawa["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "Ottawa", ax = ax[1])
ts_warped_ed["TOTAL_PRECIPITATION_EDMONTON"].plot(label = "Edmonton", ax = ax[1])
_ = ax[1].set_title("Total precipitation, warped")

plt.show()
plt.close("all")

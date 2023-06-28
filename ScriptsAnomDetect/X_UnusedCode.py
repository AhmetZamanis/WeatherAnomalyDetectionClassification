from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
model_gp = GaussianProcessFilter(n_restarts_optimizer = 50, random_state = 1923)
gp_edmonton_temp = model_gp.filter(ts_edmonton["MEAN_TEMPERATURE_EDMONTON"])



# Visualize the alignment (BROKEN)
alignment.plot_alignment(
  series2_y_offset = 25, components = ["MEAN_TEMPERATURE_OTTAWA", "MEAN_TEMPERATURE_OTTAWA"])
_ = plt.gca().set_title("Alignment, mean temperature, series 2 offset = 50")
plt.show()
plt.close("all")

alignment.plot_alignment(
  series2_y_offset = 25, components = ["TOTAL_PRECIPITATION_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"])
_ = plt.gca().set_title("Alignment, total precipitation, series 2 offset = 100")
plt.show()
plt.close("all")

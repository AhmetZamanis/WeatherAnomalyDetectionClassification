from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
model_gp = GaussianProcessFilter(n_restarts_optimizer = 50, random_state = 1923)
gp_edmonton_temp = model_gp.filter(ts_edmonton["MEAN_TEMPERATURE_EDMONTON"])



from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
model_gp = GaussianProcessFilter(n_restarts_optimizer = 50, random_state = 1923)
gp_edmonton_temp = model_gp.filter(ts_edmonton["MEAN_TEMPERATURE_EDMONTON"])


# # Get dataframes where rows = "ottawa19", "ottawa18", cols = Time, Temp, Precip (+25 for ottawa18)
# y_offset = 100
# df_alignment = []
# for pair in path:
#   idx1 = int(pair[0])
#   idx2 = int(pair[1])
#   # t1 = idx1 + 1 
#   # t2 = idx2 + 1
#   temp1 = ts_ottawa19["MEAN_TEMPERATURE_OTTAWA"][idx1].univariate_values()[0]
#   temp2 = ts_ottawa18["MEAN_TEMPERATURE_OTTAWA"][idx2].univariate_values()[0] + y_offset
#   precip1 = ts_ottawa19["TOTAL_PRECIPITATION_OTTAWA"][idx1].univariate_values()[0]
#   precip2 = ts_ottawa18["TOTAL_PRECIPITATION_OTTAWA"][idx2].univariate_values()[0] + y_offset
#   
#   data = pd.DataFrame({
#     "Time": [idx1, idx2],
#     "MeanTemp": [temp1, temp2],
#     "TotalPrecip": [precip1, precip2]
#     }
#   )
#   df_alignment.append(data)
# 
# 
# # Plot the alignments
# fig, ax = plt.subplots(2)
# 
# # Mean temp
# (ts_ottawa18["MEAN_TEMPERATURE_OTTAWA"] + y_offset).plot(
#   label = "2018", ax = ax[0], marker = "o")
# ts_ottawa19["MEAN_TEMPERATURE_OTTAWA"].plot(label = "2019", ax = ax[0], marker = "o")
# for pair in df_alignment:
#   _ = ax[0].plot("Time", "MeanTemp", data = pair, c = "black", lw = 0.75)
# 
# _ = ax[0].set_title("Alignment plot, mean temperature Ottawa")
# _ = ax[0].set_ylabel("Mean temp. (+100 for 2018)")
# _ = ax[0].set_xlabel("Day of december")
# 
# # Total precip
# (ts_ottawa18["TOTAL_PRECIPITATION_OTTAWA"] + y_offset).plot(
#   label = "2018", ax = ax[1], marker = "o")
# ts_ottawa19["TOTAL_PRECIPITATION_OTTAWA"].plot(label = "2019", ax = ax[1], marker = "o")
# for pair in df_alignment:
#   _ = ax[1].plot("Time", "TotalPrecip", data = pair, c = "black", lw = 0.75)
# 
# _ = ax[1].set_title("Alignment plot, total precipitation Ottawa")
# _ = ax[1].set_ylabel("Total precip. (+100 for 2018)")
# _ = ax[1].set_xlabel("Day of december")
# 
# plt.show()
# plt.close("all")


# # To html and back
# from IPython.display import IFrame
# fig.write_html("./HtmlPlot/plot.html", include_plotlyjs = "cdn")
# IFrame(src="./HtmlPlot/plot.html", width=700, height=600)




# # Tunable autoencoder layers
# 
# # Define hyperparameters
#     self.input_size = hyperparams_dict["input_size"]
#     self.hidden_size = hyperparams_dict["hidden_size"]
#     self.latent_size = hyperparams_dict["latent_size"]
#     self.learning_rate = hyperparams_dict["learning_rate"]
#     self.dropout = hyperparams_dict["dropout"]
#     
#     # Define architecture 
#     
#     # Encoder
#     self.encoder = torch.nn.Sequential(
#       torch.nn.Linear(self.input_size, self.hidden_size), # Hidden 1
#       torch.nn.SELU(), # Activation 1
#       torch.nn.AlphaDropout(self.dropout), # Dropout 1
#       torch.nn.Linear(self.hidden_size, self.latent_size), # Hidden 2
#       torch.nn.SELU(), # Activation 2
#       torch.nn.AlphaDropout(self.dropout) # Dropout 2
#     )
#     
#     # Decoder
#     self.decoder = torch.nn.Sequential(
#       torch.nn.Linear(self.latent_size, self.hidden_size), # Hidden 1
#       torch.nn.SELU(), # Activation 1
#       torch.nn.AlphaDropout(self.dropout), # Dropout 1
#       torch.nn.Linear(self.hidden_size, self.input_size) # Hidden 2 (output, no activation)
#     )


# hidden_size = trial.suggest_int("hidden_size", 2, 6, step = 2)
# latent_size = trial.suggest_int("latent_size", 1, (hidden_size - 1))

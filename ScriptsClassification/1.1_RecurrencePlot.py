# Canadian weather data - TS classification recurrence plot transformation
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from pyts.image import RecurrencePlot


# Create RecurrencePlot transformer
trafo_image = RecurrencePlot()


# Define function to transform each dimension and stack them, yielding
# an output of (n_seq, n_dims, seq_length, seq_length)
def trafo_multichannel(x, trafo):
  
  # Retrieve number of dimensions
  n_dims = x.shape[1]
  
  # Retrieve each dimension in a list
  dims = [x[:, i, :] for i in range(0, n_dims)]
  
  # Retrieve list of transformed channels
  channels = [trafo.fit_transform(dims[i]) for i in range(0, n_dims)]
  
  # Stack the outputs
  output = np.stack(channels, axis = 1)
  
  return output
  
  
# Transform the features
x_train = trafo_multichannel(x_train, trafo_image)
x_test = trafo_multichannel(x_test, trafo_image)


# The output for 1 sequence is of shape (n_dims, seq_length, seq_length). Each
# channel is the values of the recurrence plot for one dimension. We have 8 images
# of 28x28, per 28-day period, per city.
x_train[0].shape


# Plot the recurrence plot for two consecutive sequences per city, for the first
# dimension. The plot for each sequence is the pairwise similarity matrix of each 
# trajectory in that sequence. The resulting "images" should identify the city 
# when compared with other "images" for the same city.
# The plots for the time dimensions in a period are the same for all cities, as
# expected.
fig, ax = plt.subplots(3, 2, sharex = True, sharey = True)
_ = plt.suptitle("Recurrence plots of first two 28-day periods in the data, first dimension (mean temp)")

_ = ax[0, 0].imshow(x_train[0][0], cmap = "binary", origin = "lower")
_ = ax[0, 0].set_title("Ottawa")

_ = ax[1, 0].imshow(x_train[834][0], cmap = "binary", origin = "lower")
_ = ax[1, 0].set_title("Toronto")

_ = ax[2, 0].imshow(x_train[1668][0], cmap = "binary", origin = "lower")
_ = ax[2, 0].set_title("Vancouver")

_ = ax[0, 1].imshow(x_train[1][0], cmap = "binary", origin = "lower")
_ = ax[0, 1].set_title("Ottawa")

_ = ax[1, 1].imshow(x_train[835][0], cmap = "binary", origin = "lower")
_ = ax[1, 1].set_title("Toronto")

_ = ax[2, 1].imshow(x_train[1669][0], cmap = "binary", origin = "lower")
_ = ax[2, 1].set_title("Vancouver")
plt.show()
plt.close("all")


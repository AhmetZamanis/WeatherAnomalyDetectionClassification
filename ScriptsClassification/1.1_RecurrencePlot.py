# Canadian weather data - TS classification recurrence plot transformation
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from pyts.image import RecurrencePlot
# from sklearn.preprocessing import MinMaxScaler


# Create RecurrencePlot transformer
trafo_image = RecurrencePlot()


# # Scaling over one dimension: Done with the minimums and maximums across all
# # subsequences, all locations of the images
# transformed = trafo_image.fit_transform(x_train[:, 0, :])
# dim_min = np.min(transformed)
# dim_max = np.max(transformed)
# transformed_scaled = (transformed - dim_min) / (dim_max - dim_min)
# x_train[:, 0, :].shape
# transformed.shape


# Define function to transform each dimension into images & scale & stack them, 
# yielding an output of (n_seq, n_dims, seq_length, seq_length)
def get_images(x_train, x_test, trafo):
  
  # Retrieve number of dimensions
  n_dims = x.shape[1]
  
  # Retrieve each dimension in a list
  dims_train = [x_train[:, i, :] for i in range(0, n_dims)]
  dims_test = [x_test[:, i, :] for i in range(0, n_dims)]
  
  # Fit & transform training data for each dimension
  fitted_trafos = []
  fitted_mins = []
  fitted_maxi = []
  channels_train = []
  channels_test = []
  for dim in dims_train:
    
    # Copy base transformer
    fitted_trafo = trafo
    
    # Fit & transform dimension
    transformed = fitted_trafo.fit_transform(dim)
    
    # Save fitted transformer
    fitted_trafos.append(fitted_trafo)
    
    # Retrieve & save min, max of the dimension matrix
    dim_min = np.min(transformed)
    dim_max = np.max(transformed)
    fitted_mins.append(dim_min)
    fitted_maxi.append(dim_max)
    
    # Scale values between 0 and 1, append to list
    transformed_scaled = (transformed - dim_min) / (dim_max - dim_min)
    channels_train.append(transformed_scaled)
    
  # Transform testing data for each dimension
  for i, dim in enumerate(dims_test):
    
    # Transform & scale dimension, append to list
    transformed = fitted_trafos[i].transform(dim)
    transformed_scaled = (transformed - fitted_mins[i]) / (fitted_maxi[i] - fitted_mins[i])
    channels_test.append(transformed_scaled)
    
  # Stack the outputs
  output_train = np.stack(channels_train, axis = 1)
  output_test = np.stack(channels_test, axis = 1)

  return output_train, output_test


# Split validation data from training data

# Get indices
l = len(y_train)
len_val = int(l / 3 * 0.2)
len_tr = int(l / 3 - len_val)
j = int(l / 3)
idx_tr = list(range(0, len_tr)) + list(range(j, len_tr + j)) + list(range(j * 2, len_tr + (j * 2)))
idx_val = list(range(0, l))
idx_val = list(set(idx_val).difference(idx_tr))

# Perform split
y_tr, y_val = y_train[idx_tr], y_train[idx_val]
x_tr, x_val = x_train[idx_tr], x_train[idx_val]

  
# Transform the features
x_train, x_test = get_images(x_train, x_test, trafo_image)
x_tr, x_val = get_images(x_tr, x_val, trafo_image)


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
_ = plt.suptitle("Recurrence plots of first two 28-day periods in the data, second dimension (total precipitation)")

_ = ax[0, 0].imshow(x_train[0][1], cmap = "binary", origin = "lower")
_ = ax[0, 0].set_title("Ottawa")

_ = ax[1, 0].imshow(x_train[834][1], cmap = "binary", origin = "lower")
_ = ax[1, 0].set_title("Toronto")

_ = ax[2, 0].imshow(x_train[1668][1], cmap = "binary", origin = "lower")
_ = ax[2, 0].set_title("Vancouver")

_ = ax[0, 1].imshow(x_train[1][1], cmap = "binary", origin = "lower")
_ = ax[0, 1].set_title("Ottawa")

_ = ax[1, 1].imshow(x_train[835][1], cmap = "binary", origin = "lower")
_ = ax[1, 1].set_title("Toronto")

_ = ax[2, 1].imshow(x_train[1669][1], cmap = "binary", origin = "lower")
_ = ax[2, 1].set_title("Vancouver")
plt.show()
plt.close("all")


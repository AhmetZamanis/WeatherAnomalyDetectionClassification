# Canadian weather data - TS classification recurrence plot transformation
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from pyts.image import RecurrencePlot
from X_HelperFunctionsClassif import get_images, plot_images


# Create RecurrencePlot transformer
trafo_image = RecurrencePlot()


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


# Plot the recurrence plot for two consecutive sequences per city, for the weather
# dimensions. The plot for each sequence is the pairwise similarity matrix of each 
# trajectory in that sequence. The resulting "images" should identify the city 
# when compared with other "images" for the same city.
plot_images(x_train, 0, "MeanTemp", 0, 1, j)
plot_images(x_train, 1, "TotalPrecip", 0, 1, j)


# The plots for the time dimensions in a period are the same for all cities, as
# expected.
plot_images(x_train, 2, "MonthSin", 0, 1, j)
plot_images(x_train, 3, "MonthCos", 0, 1, j)
plot_images(x_train, 4, "WeekSin", 0, 1, j)
plot_images(x_train, 5, "WeekCos", 0, 1, j)
plot_images(x_train, 6, "DaySin", 0, 1, j)
plot_images(x_train, 7, "DayCos", 0, 1, j)


# Canadian weather data - TS classification with joint recurrence + NN
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from pyts.multivariate.image import JointRecurrencePlot


# Transform features into images with JointRecurrence
trafo_image = JointRecurrencePlot()


# The transformed output is of shape (n_sequences, seq_length, seq_length)
x_train = trafo_image.fit_transform(x_train) 
x_test = trafo_image.transform(x_test)

# The output for 1 sequence is of shape (seq_length, seq_length), the matrix of
# values of the joint recurrence plot, which is obtained as the Hadamard product 
# of recurrence plots for each dimension. This is a single-channel input in terms
# of CNN terminology (29 x 29 x 1). Applying the recurrence plot to each dimension 
# separately would give us a 8-dimensional channel (29 x 29 x 8).
# (taking their Hadamard is kind of similar to running them through a multi-channel CNN?).
x_train[0].shape


# Plot the recurrence plot for two consecutive sequences per city
# The plot for each sequence is the pairwise similarity matrix of each trajectory 
# in that sequence. The resulting "image" should identify the city when compared 
# with other "images" for the same city.
fig, ax = plt.subplots(3, 2, sharex = True, sharey = True)
_ = plt.suptitle("Joint recurrence plots of first 2 months in the data")

_ = ax[0, 0].imshow(x_train[0], cmap = "binary", origin = "lower")
_ = ax[0, 0].set_title("Ottawa")

_ = ax[1, 0].imshow(x_train[804], cmap = "binary", origin = "lower")
_ = ax[1, 0].set_title("Toronto")

_ = ax[2, 0].imshow(x_train[1610], cmap = "binary", origin = "lower")
_ = ax[2, 0].set_title("Vancouver")

_ = ax[0, 1].imshow(x_train[1], cmap = "binary", origin = "lower")
_ = ax[0, 1].set_title("Ottawa")

_ = ax[1, 1].imshow(x_train[805], cmap = "binary", origin = "lower")
_ = ax[1, 1].set_title("Toronto")

_ = ax[2, 1].imshow(x_train[1611], cmap = "binary", origin = "lower")
_ = ax[2, 1].set_title("Vancouver")
plt.show()
plt.close("all")




# # Calculate multiclass performance metrics
# 
# # Accuracy
# accuracy_score(y_test, preds_muse)
# 
# # Log loss
# log_loss(y_test, probs_muse, labels = classes)
# 
# 
# # Plot confusion matrix
# plot_confusion(y_test, preds_muse, classes, "WEASELMUSE + logistic classifier")


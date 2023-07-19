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

# The output for 1 sequence is of shape (seq_length, seq_length), the X and Y
# values of the joint recurrence plot, which is obtained as the Hadamard product 
# of recurrence plots for each dimension.
x_train[0].shape

y_train[1609]


# Plot the recurrence plot for one sequence per city
fig, ax = plt.subplots(3, sharex = True)
_ = plt.suptitle("Joint recurrence plots of first month in the data")

_ = ax[0].imshow(x_train[0], cmap = "binary", origin = "lower")
_ = ax[0].set_title("Ottawa")

_ = ax[1].imshow(x_train[804], cmap = "binary", origin = "lower")
_ = ax[1].set_title("Toronto")

_ = ax[2].imshow(x_train[1610], cmap = "binary", origin = "lower")
_ = ax[2].set_title("Vancouver")
plt.show()
plt.close("all")




# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_muse)

# Log loss
log_loss(y_test, probs_muse, labels = classes)


# Plot confusion matrix
plot_confusion(y_test, preds_muse, classes, "WEASELMUSE + logistic classifier")


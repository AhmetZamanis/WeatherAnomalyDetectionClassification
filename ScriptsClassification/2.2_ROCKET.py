# Canadian weather data - TS classification with ROCKET + Ridge regression
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())

# from sktime.transformations.compose import ColumnwiseTransformer
# from sklearn.preprocessing import MinMaxScaler
from sktime.classification.kernel_based import RocketClassifier


# Create RocketClassifier
model_rocket = RocketClassifier(
  use_multivariate = "yes", n_jobs = -1, random_state = 1923)


# Fit on training data
_ = model_rocket.fit(x_train, y_train)


# Predict testing data
preds_rocket = model_rocket.predict(x_test)
probs_rocket = model_rocket.predict_proba(x_test)


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_rocket)

# Log loss
log_loss(y_test, probs_rocket, labels = classes)


# Plot confusion matrix
plot_confusion(y_test, preds_rocket, classes, "Rocket + Ridge classifier")


# Canadian weather data - TS classification with ROCKET + Ridge regression
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


# from sktime.transformations.panel.rocket import Rocket
from sktime.classification.kernel_based import RocketClassifier, Arsenal


# # Perform Rocket transformation and view outputs
# trafo_rocket = Rocket(n_jobs = -1, random_state = 1923)
# x_rocket = trafo_rocket.fit_transform(x)
# x_rocket.shape # 20k columns (2 features per 10k kernels) per observation
# trafo_rocket.kernels # Kernel parameters


# Create RocketClassifier
model_rocket = RocketClassifier(
  use_multivariate = "yes", n_jobs = -1, random_state = 1923)


# Test classifier
preds_rocket, probs_rocket, acc_rocket, loss_rocket = test_model(
  model_rocket, x_train, x_test, y_train, y_test, scale = False)


# View metrics
acc_rocket # 0.5400641025641025
loss_rocket # 16.57777006839202


# Plot confusion matrix
plot_confusion(y_test, preds_rocket, classes, "Rocket + Ridge classifier")




# Create Arsenal classifier (probabilistic ROCKET ensemble, memory intensive)
model_arsenal = Arsenal(random_state = 1923)


# Test classifier
preds_arsenal, probs_arsenal, acc_arsenal, loss_arsenal = test_model(
  model_arsenal, x_train, x_test, y_train, y_test, scale = False)


# View metrics
acc_arsenal # 0.5576923076923077
loss_arsenal # 2.3082290412092235


# Plot confusion matrix
plot_confusion(y_test, preds_arsenal, classes, "Arsenal + Ridge classifier")

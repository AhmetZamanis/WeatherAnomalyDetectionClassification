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
# In sktime, Multivariate Rocket is the same as Univariate Rocket
# Minirocket and Multirocket are almost deterministic versions of Rocket, and the
# latter also derives more features from each kernel. Both have univariate & 
# multivariate versions to choose from
# All of them have binary predicted probabilities
model_rocket = RocketClassifier(
  rocket_transform = "multirocket", use_multivariate = "yes", n_jobs = -1, 
  random_state = 1923)


# Test classifier
preds_rocket, probs_rocket, acc_rocket, loss_rocket = test_model(
  model_rocket, x_train, x_test, y_train, y_test, scale = False)


# View metrics
acc_rocket # ROCKET: 0.54, MINIROCKET: 0.68, MULTIROCKET: 0.67
loss_rocket # ROCKET: 16.57, MINIROCKET: 11.32, MULTIROCKET: 11.72


# Plot confusion matrix
plot_confusion(y_test, preds_rocket, classes, "Rocket + Ridge classifier")




# Create Arsenal classifier (probabilistic ROCKET ensemble, memory intensive)
# Default Rocket is same for univariate & multivariate
# Mini & multirocket are automatically chosen to be univariate or multivariate
# based on input dimensions
model_arsenal = Arsenal(rocket_transform = "multirocket", random_state = 1923)


# Test classifier
preds_arsenal, probs_arsenal, acc_arsenal, loss_arsenal = test_model(
  model_arsenal, x_train, x_test, y_train, y_test, scale = False)


# View metrics
acc_arsenal # ROCKET: 0.55, MINIROCKET: 0.73, MULTIROCKET: 0.75
loss_arsenal # ROCKET: 2.30, MINIROCKET: 1.93, MULTIROCKET: 0.75


# Plot confusion matrix
plot_confusion(y_test, preds_arsenal, classes, "Arsenal + Ridge classifier")

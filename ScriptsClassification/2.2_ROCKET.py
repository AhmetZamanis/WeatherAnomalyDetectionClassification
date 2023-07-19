# Canadian weather data - TS classification with ROCKET + Ridge regression
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


# from sktime.transformations.panel.rocket import Rocket
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.kernel_based import Arsenal


# # Perform Rocket transformation and view outputs
# trafo_rocket = Rocket(n_jobs = -1, random_state = 1923)
# x_rocket = trafo_rocket.fit_transform(x)
# x_rocket.shape # 20k columns (2 features per 10k kernels) per observation
# trafo_rocket.kernels # Kernel parameters


# Create RocketClassifier
model_rocket = RocketClassifier(
  use_multivariate = "yes", n_jobs = -1, random_state = 1923)


# Fit on training data
_ = model_rocket.fit(x_train, y_train)


# Predict testing data
preds_rocket = model_rocket.predict(x_test)
probs_rocket = model_rocket.predict_proba(x_test) # ROCKET is not probabilistic


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_rocket)

# Log loss
log_loss(y_test, probs_rocket, labels = classes) # ROCKET is not probabilistic


# Plot confusion matrix
plot_confusion(y_test, preds_rocket, classes, "Rocket + Ridge classifier")




# Create Arsenal classifier (probabilistic ROCKET ensemble, memory intensive)
model_arsenal = Arsenal(random_state = 1923)


# Fit on training data
_ = model_arsenal.fit(x_train, y_train)


# Predict testing data
preds_arsenal = model_arsenal.predict(x_test)
probs_arsenal = model_arsenal.predict_proba(x_test) 


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_arsenal)

# Log loss
log_loss(y_test, probs_arsenal, labels = classes) 


# Plot confusion matrix
plot_confusion(y_test, preds_arsenal, classes, "Arsenal + Ridge classifier")

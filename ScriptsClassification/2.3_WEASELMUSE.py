# Canadian weather data - TS classification with WEASELMUSE + logistic regression
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from sktime.classification.dictionary_based import MUSE


# Create WEASELMUSE classifier
model_muse = MUSE(
  use_first_order_differences = False, # Not meaningful for time features
  support_probabilities = True, # Train LogisticRegression which outputs probs.
  n_jobs = -1, random_state = 1923)


# Fit on training data
_ = model_muse.fit(x_train, y_train)


# Predict testing data
preds_muse = model_muse.predict(x_test)
probs_muse = model_muse.predict_proba(x_test)


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_muse)

# Log loss
log_loss(y_test, probs_muse, labels = classes)


# Plot confusion matrix
plot_confusion(y_test, preds_muse, classes, "WEASELMUSE + logistic classifier")


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


# Test classifier
preds_muse, probs_muse, acc_muse, loss_muse = test_model(
  model_muse, x_train, x_test, y_train, y_test, scale = True)


# View metrics
acc_muse # 0.6073717948717948
loss_muse # 1.1958264093104147


# Plot confusion matrix
plot_confusion(y_test, preds_muse, classes, "WEASELMUSE + logistic classifier")


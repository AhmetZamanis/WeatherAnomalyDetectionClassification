# Canadian weather data - TS classification with KNN algorithm
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())


from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


# Create KNN classifier
model_knn = KNeighborsTimeSeriesClassifier(n_neighbors = 3, n_jobs = -1)


# Test classifier
preds_knn, probs_knn, acc_knn, loss_knn = test_model(
  model_knn, x_train, x_test, y_train, y_test, scale = True)


# View metrics
acc_knn # 0.6602564102564102
loss_knn # 5.575000713123325


# Plot confusion matrix
plot_confusion(y_test, preds_knn, classes, "KNN with DTW distance")


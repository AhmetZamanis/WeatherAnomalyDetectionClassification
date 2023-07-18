# Canadian weather data - TS classification with KNN algorithm
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1_DataPrep.py").read())

# from sktime.transformations.compose import ColumnwiseTransformer
# from sklearn.preprocessing import MinMaxScaler
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


# Create KNN classifier
model_knn = KNeighborsTimeSeriesClassifier(n_neighbors = 3, n_jobs = -1)


# # Scale data (wrapper doesn't work)
# scaler = MinMaxScaler(feature_range = (0, 1))
# scaler = ColumnwiseTransformer(scaler)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# Fit on training data
_ = model_knn.fit(x_train, y_train)


# Predict testing data
preds_knn = model_knn.predict(x_test)
probs_knn = model_knn.predict_proba(x_test)


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_knn)

# Log loss
log_loss(y_test, probs_knn, labels = classes)


# Plot confusion matrix
plot_confusion(y_test, preds_knn, classes, "KNN with DTW distance")


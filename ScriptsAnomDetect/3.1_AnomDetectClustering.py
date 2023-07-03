# Canadian weather data - TS anomaly detection with clustering algorithms
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from darts.ad.scorers.kmeans_scorer import KMeansScorer


# Create K-means scorer
scorer = KMeansScorer(window = 1, k = 12, random_state = 1923)


# Perform anomaly scoring
scores_train, scores_test, scores = score(ts_train, ts_test, scorer, scaler)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot scores & original series
plot_series("K-means scorer", ts_train, ts_test, scores_train, scores_test)


# Plot distributions of anomaly scores
plot_dist("K-means scorer", scores_train, scores_test)


# 3D anomalies plot
plot_anom3d("K-means scorer", ts_ottawa, anoms)


# Detections plot
plot_detection("K-means scores", q, ts_ottawa, scores, anoms)


# Special to scorer: Clustering plot
train_labels = scorer.model.labels_.astype(str)
fig = px.scatter_3d(
  x = ts_train['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_train['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_train.time_index.month,
  color = train_labels,
  title = "K-Means clustering plot, train set",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Clusters"}
)
fig.show()


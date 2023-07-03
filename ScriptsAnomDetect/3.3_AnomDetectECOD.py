# Canadian weather data - TS anomaly detection with ECOD model
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.ecod import ECOD


# Create ECOD scorer
ecod = ECOD(contamination = 0.01)
scorer = PyODScorer(model = ecod, window = 1)
scorer_name = "ECOD scorer"


# Perform anomaly scoring
scores_train, scores_test, scores = score(ts_train, ts_test, scorer)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot scores & original series
plot_series(scorer_name, ts_train, ts_test, scores_train, scores_test)


# Plot distributions of anomaly scores
plot_dist(scorer_name, scores_train, scores_test)


# 3D anomalies plot
plot_anom3d(scorer_name, ts_ottawa, anoms)


# Detections plot
plot_detection("ECOD scores", q, ts_ottawa, scores, anoms)
# ECOD tends to falsely flag very hot and very cold days as anomalies, regardless 
# of the month. Probably not good for multivariate anomaly detection with interactions.
# It still finds most precipitation anomalies.


# Special to scorer: Explaining a single training point's outlier scoring
idx_max_precip = np.argmax(ts_train['TOTAL_PRECIPITATION_OTTAWA'].univariate_values())
scorer.model.explain_outlier(
  ind = idx_max_precip, # Index of point to explain
  columns = [1, 2, 3, 4], # Dimensions to explain (variables + month cyclical)
  feature_names = ["MeanTemp", "TotalPrecip", "MonthSin", "MonthCos"]
  )
plt.close("all")
# Even the highest precipitation training sample was flagged mainly due to its
# temperature.

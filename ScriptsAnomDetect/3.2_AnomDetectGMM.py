# Canadian weather data - TS anomaly detection with Gaussian Mixture Model
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.gmm import GMM


# Create GMM scorer
gmm = GMM(
  n_components = 4, # N. of Gaussian mixture components
  n_init = 10, # N. of initializations for expectation maximization
  contamination = 0.01, # % of expected anomalies in the dataset
  random_state = 1923
)
scorer = PyODScorer(model = gmm, window = 1)
scorer_name = "GMM scorer"


# Perform anomaly scoring
scores_train, scores_test, scores = score(ts_train, ts_test, scorer, scaler)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot scores & original series
plot_series(scorer_name, ts_train, ts_test, scores_train, scores_test)


# Plot distributions of anomaly scores
plot_dist(scorer_name, scores_train, scores_test)


# 3D anomalies plot
plot_anom3d(scorer_name, ts_ottawa, anoms)


# Detections plot
plot_detection("GMM scores", q, ts_ottawa, scores, anoms)

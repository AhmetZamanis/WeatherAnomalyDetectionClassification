# Canadian weather data - TS anomaly detection with Isolation Forest model
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.iforest import IForest


# Create IForest scorer
iforest = IForest(
  n_estimators= 500,
  contamination = 0.01,
  random_state = 1923
  )
scorer = PyODScorer(model = iforest, window = 1)
scorer_name = "IForest scorer"


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
plot_detection("PCA scores", q, ts_ottawa, scores, anoms)
# The binary detections seem just as well as PCAs, but the scores are not as clearly
# discriminative.


# Feature importances (can be misleading for high cardinality features, i.e. day
# and week features)
feat_imp = pd.DataFrame({
  "Feature importance": scorer.model.feature_importances_,
  "Feature": ts_ottawa.components
}).sort_values("Feature importance", ascending = False)
_ = sns.barplot(data = feat_imp, x = "Feature importance", y = "Feature")
plt.show()
plt.close("all")


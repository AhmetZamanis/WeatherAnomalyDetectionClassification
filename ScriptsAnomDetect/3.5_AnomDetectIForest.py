# Canadian weather data - TS anomaly detection with Isolation Forest model
# Data source: https://openml.org/search?type=data&status=active&id=43843

exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.iforest import IForest


# Create IForest scorer
iforest = IForest(
  n_estimators= 500,
  contamination = 0.01,
  random_state = 1923
  )
scorer = PyODScorer(model = iforest, window = 1)


# Perform anomaly scoring
scores_train, scores_test, scores = score(ts_train, ts_test, scorer)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot anomaly scores
fig, ax = plt.subplots(3, sharex = True)

# Anomaly scores
scores_train.plot(ax = ax[0])
scores_test.plot(ax = ax[0])
_ = ax[0].set_title("Anomaly scores, IForest Model")
# _ = ax[0].set_ylim(0)

# MeanTemp
ts_train['MEAN_TEMPERATURE_OTTAWA'].plot(ax = ax[1])
ts_test['MEAN_TEMPERATURE_OTTAWA'].plot(ax = ax[1])
_ = ax[1].set_title("Mean temperatures")

# TotalPrecip
ts_train['TOTAL_PRECIPITATION_OTTAWA'].plot(label = "Train set", ax = ax[2])
ts_test['TOTAL_PRECIPITATION_OTTAWA'].plot(label = "Test set", ax = ax[2])
_ = ax[2].set_title("Total precipitation")

plt.show()
plt.close("all")


# Plot distributions of anomaly scores
df_scores = scores_train.pd_dataframe().rename({"0": "Train"}, axis = 1)
df_scores["Test"] = scores_test.values()[0:-1]
df_scores = df_scores.melt(var_name = "Set", value_name = "Anomaly scores")
_ = sns.kdeplot(data = df_scores, x = "Anomaly scores", hue = "Set")
_ = plt.title("Distributions of IForest anomaly scores")
plt.show()
plt.close("all")


# Retrieve dates, variables and anomaly labels in dataframes, separately for
# positive and negative observations
df_positive = pd.DataFrame({
  "Date": ts_ottawa.time_index,
  "IForest score": np.where(
    anoms.univariate_values() == 1, 
    scores.univariate_values(), np.nan),
  "Mean temperature": np.where(
    anoms.univariate_values() == 1, 
    ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(), np.nan),
  "Total precipitation": np.where(
    anoms.univariate_values() == 1, 
    ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(), np.nan)
  }
)

df_negative = pd.DataFrame({
  "Date": ts_ottawa.time_index,
  "IForest score": np.where(
    anoms.univariate_values() == 0, 
    scores.univariate_values(), np.nan),
  "Mean temperature": np.where(
    anoms.univariate_values() == 0, 
    ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(), np.nan),
  "Total precipitation": np.where(
    anoms.univariate_values() == 0, 
    ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(), np.nan)
  }
)


# Plot original series, colored by anomalous & non-anomalous
fig, ax = plt.subplots(3, sharex = True)
_ = fig.suptitle("Anomaly detections with 99th percentile IForest scores\nBlue = Anomalous days")

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "IForest score", ax = ax[0])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "IForest score", ax = ax[0])

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "Mean temperature", ax = ax[1])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "Mean temperature", ax = ax[1])

_ = sns.lineplot(data = df_negative, x = "Date", y = "Total precipitation", ax = ax[2])
_ = sns.lineplot(data = df_positive, x = "Date", y = "Total precipitation", ax = ax[2])

plt.show()
plt.close("all")
# The scores themselves are small and appear non-discriminatory, but similar to PCA,
# IForest exclusively seems to pick hot & rainy days as anomalies


# 3D anomalies plot
fig = px.scatter_3d(
  x = ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_ottawa.time_index.month,
  color = anoms.univariate_values().astype(str),
  title = "IForest anomaly labeling",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Anomaly flag"}
)
fig.show()


# Feature importances (can be misleading for high cardinality features, i.e. day
# and week features)
feat_imp = pd.DataFrame({
  "Feature importance": scorer.model.feature_importances_,
  "Feature": ts_ottawa.components
}).sort_values("Feature importance", ascending = False)
_ = sns.barplot(data = feat_imp, x = "Feature importance", y = "Feature")
plt.show()
plt.close("all")


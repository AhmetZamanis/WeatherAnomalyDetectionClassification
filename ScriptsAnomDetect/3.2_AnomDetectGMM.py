# Canadian weather data - TS anomaly detection with Gaussian Mixture Model
# Data source: https://openml.org/search?type=data&status=active&id=43843

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


# Perform anomaly scoring
scores_train, scores_test, scores = score(ts_train, ts_test, scorer, scaler)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot anomaly scores
fig, ax = plt.subplots(3, sharex = True)

# Anomaly scores
scores_train.plot(ax = ax[0])
scores_test.plot(ax = ax[0])
_ = ax[0].set_title("Anomaly scores, Gaussian Mixture Model")
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
_ = plt.title("Distributions of GMM anomaly scores")
plt.show()
plt.close("all")


# Retrieve dates, variables and anomaly labels in dataframes, separately for
# positive and negative observations
df_positive = pd.DataFrame({
  "Date": ts_ottawa.time_index,
  "GMM score": np.where(
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
  "GMM score": np.where(
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
_ = fig.suptitle("Anomaly detections with 99th percentile GMM scores\nBlue = Anomalous days")

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "GMM score", ax = ax[0])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "GMM score", ax = ax[0])

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "Mean temperature", ax = ax[1])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "Mean temperature", ax = ax[1])

_ = sns.lineplot(data = df_negative, x = "Date", y = "Total precipitation", ax = ax[2])
_ = sns.lineplot(data = df_positive, x = "Date", y = "Total precipitation", ax = ax[2])

plt.show()
plt.close("all")


# 3D anomalies plot
fig = px.scatter_3d(
  x = ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_ottawa.time_index.month,
  color = anoms.univariate_values().astype(str),
  title = "GMM anomaly labeling",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Anomaly flag"}
)
fig.show()

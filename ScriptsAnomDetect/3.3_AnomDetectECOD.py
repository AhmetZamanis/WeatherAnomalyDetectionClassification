# Canadian weather data - TS anomaly detection with ECOD model
# Data source: https://openml.org/search?type=data&status=active&id=43843

exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


# from sklearn.preprocessing import MinMaxScaler
# from darts.dataprocessing.transformers.scaler import Scaler
from pyod.models.ecod import ECOD
from darts.ad.scorers.pyod_scorer import PyODScorer
from darts.ad.detectors.quantile_detector import QuantileDetector


# Concatenate time covariates to Ottawa temperature series
ts_ottawa = ts_ottawa.concatenate(ts_covars, axis = 1)


# Split train-test: Before vs. after 1980
ts_train = ts_ottawa.drop_after(pd.Timestamp("1980-01-01"))
ts_test = ts_ottawa.drop_before(pd.Timestamp("1979-12-31"))


# # Scale series between -1 and 1
# scaler = Scaler(MinMaxScaler(feature_range = (-1, 1)))
# ts_train = scaler.fit_transform(ts_train)
# ts_test = scaler.transform(ts_test)


# Fit ECOD scorer on train set
model_ecod = ECOD(contamination = 0.01)
scorer_ecod = PyODScorer(model = model_ecod, window = 1)
_ = scorer_ecod.fit(ts_train)
scores_train = scorer_ecod.score(ts_train)
scores_test = scorer_ecod.score(ts_test)
scores = scores_train.append(scores_test)


# Plot anomaly scores
fig, ax = plt.subplots(3, sharex = True)

# Anomaly scores
scores_train.plot(ax = ax[0])
scores_test.plot(ax = ax[0])
_ = ax[0].set_title("Anomaly scores, ECOD Model")
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
_ = plt.title("Distributions of ECOD anomaly scores")
plt.show()
plt.close("all")


# Quantile anomaly detection
detector = QuantileDetector(high_quantile = 0.99)
_ = detector.fit(scores_train)
anoms_train = detector.detect(scores_train)
anoms_test = detector.detect(scores_test)
anoms = anoms_train.append(anoms_test)


# Retrieve dates, variables and anomaly labels in dataframes, separately for
# positive and negative observations
df_positive = pd.DataFrame({
  "Date": ts_ottawa.time_index,
  "ECOD score": np.where(
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
  "ECOD score": np.where(
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
_ = fig.suptitle("Anomaly detections with 99th percentile ECOD scores\nBlue = Anomalous days")

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "ECOD score", ax = ax[0])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "ECOD score", ax = ax[0])

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
  title = "ECOD anomaly labeling",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Anomaly flag"}
)
fig.show()
# ECOD tends to falsely flag very hot and very cold days as anomalies, regardless 
# of the month. Probably not good for multivariate anomaly detection with interactions.
# It still finds most precipitation anomalies.


# Explaining as single training point's outlier scoring
idx_max_precip = np.argmax(ts_train['TOTAL_PRECIPITATION_OTTAWA'].univariate_values())
scorer_ecod.model.explain_outlier(
  ind = idx_max_precip, # Index of point to explain
  columns = [1, 2, 3, 4], # Dimensions to explain (variables + month cyclical)
  feature_names = ["MeanTemp", "TotalPrecip", "MonthSin", "MonthCos"]
  )
plt.close("all")
# Even the highest precipitation training sample was flagged mainly due to its
# temperature.
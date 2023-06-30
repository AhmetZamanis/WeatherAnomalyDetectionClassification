# Canadian weather data - TS anomaly detection with PCA
# Data source: https://openml.org/search?type=data&status=active&id=43843

exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers.scaler import Scaler
from pyod.models.pca import PCA
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


# Fit PCA scorer on train set
model_pca = PCA(contamination = 0.01, standardization = True, random_state = 1923)
scorer_pca = PyODScorer(model = model_pca, window = 1)
_ = scorer_pca.fit(ts_train)
scores_train = scorer_pca.score(ts_train)
scores_test = scorer_pca.score(ts_test)
scores = scores_train.append(scores_test)


# Quantile anomaly detection
detector = QuantileDetector(high_quantile = 0.99)
_ = detector.fit(scores_train)
anoms_train = detector.detect(scores_train)
anoms_test = detector.detect(scores_test)
anoms = anoms_train.append(anoms_test)


# Variances explained by each component: First 3 PCs explain almost 99%
pc_variances = scorer_pca.model.explained_variance_ratio_ * 100
pc_variances = [round(x, 2) for x in pc_variances]

# Heatplot of PC loadings, X = PCs, Y = Features's contribution to the PCs
pc_loadings = pd.DataFrame(
  scorer_pca.model.components_.T,
  columns = pc_variances,
  index = ts_ottawa.components
)
_ = sns.heatmap(pc_loadings, cmap = "PiYG")
_ = plt.title("PC loadings")
_ = plt.xlabel("% variances explained by PCs")
_ = plt.ylabel("Features")
plt.show()
plt.close("all")
# PC1 is considerably influenced by MeanTemp, Month, Week, Day
# PC2 is solely influenced by the time features
# PC3 is solely influenced by TotalPrecip
# PC4 is solely influenced by MeanTemp
# Final PCs are strongly and solely influenced by week and day features. Maybe they
# are decisive in selecting anomalies?


# Plot anomaly scores
fig, ax = plt.subplots(3, sharex = True)

# Anomaly scores
scores_train.plot(ax = ax[0])
scores_test.plot(ax = ax[0])
_ = ax[0].set_title("Anomaly scores, PCA")
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
_ = plt.title("Distributions of PCA anomaly scores")
plt.show()
plt.close("all")


# # Principal components plot
# fig = px.scatter_3d(
#   x = scorer_pca.model.components_[0],
#   y = scorer_pca.model.components[1],
#   z = scorer_pca.model.components[2],
#   color = anoms_train.univariate_values().astype(str),
#   title = "PCA components plot",
#   labels = {
#     "x": "PC1",
#     "y": "PC2",
#     "z": "PC3",
#     "color": "Anomaly labels"}
# )
# fig.show()


# 3D anomalies plot
fig = px.scatter_3d(
  x = ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_ottawa.time_index.month,
  color = anoms.univariate_values().astype(str),
  title = "PCA anomaly flags",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Anomaly flags"}
)
fig.show()
# No "mistaken" anomalies close to the centers of any month.


# Retrieve dates, variables and anomaly labels in dataframes, separately for
# positive and negative observations
df_positive = pd.DataFrame({
  "Date": ts_ottawa.time_index,
  "PCA score": np.where(
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
  "PCA score": np.where(
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
_ = fig.suptitle("Anomaly detections with 99th percentile PCA scores\nBlue = Anomalous days")

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "PCA score", ax = ax[0])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "PCA score", ax = ax[0])

_ = sns.lineplot(data = df_negative,  x = "Date",  y = "Mean temperature", ax = ax[1])
_ = sns.lineplot(data = df_positive,  x = "Date",  y = "Mean temperature", ax = ax[1])

_ = sns.lineplot(data = df_negative, x = "Date", y = "Total precipitation", ax = ax[2])
_ = sns.lineplot(data = df_positive, x = "Date", y = "Total precipitation", ax = ax[2])

plt.show()
plt.close("all")
# PCA seems to select solely hot & unusually rainy days as anomalies.


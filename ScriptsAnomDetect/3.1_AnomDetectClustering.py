# Canadian weather data - TS anomaly detection with clustering algorithms
# Data source: https://openml.org/search?type=data&status=active&id=43843

exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers.scaler import Scaler
from darts.ad.scorers.kmeans_scorer import KMeansScorer


# Concatenate time covariates to Ottawa temperature series
ts_ottawa = ts_ottawa.concatenate(ts_covars, axis = 1)


# Split train-test: Before vs. after 1980
test_start = pd.Timestamp("1980-01-01")
ts_train = ts_ottawa.drop_after(test_start)
ts_test = ts_ottawa.drop_before(test_start)


# Scale series between -1 and 1
scaler = Scaler(MinMaxScaler(feature_range = (-1, 1)))
ts_train = scaler.fit_transform(ts_train)
ts_test = scaler.transform(ts_test)


# Fit K-means scorer on train set
scorer_kmeans = KMeansScorer(window = 30, k = 12, random_state = 1923)
_ = scorer_kmeans.fit(ts_train)
scores_train = scorer_kmeans.score(ts_train)
scores_test = scorer_kmeans.score(ts_test)


# Plot anomaly scores
fig, ax = plt.subplots(3, sharex = True)

# Anomaly scores
scores_train.plot(ax = ax[0])
scores_test.plot(ax = ax[0])
_ = ax[0].set_title("Anomaly scores, K-Means")
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
df_scores["Test"] = scores_test.values()
df_scores = df_scores.melt(var_name = "Set", value_name = "Anomaly scores")
_ = sns.kdeplot(data = df_scores, x = "Anomaly scores", hue = "Set")
_ = plt.title("Distributions of K-Means anomaly scores")
plt.show()
plt.close("all")


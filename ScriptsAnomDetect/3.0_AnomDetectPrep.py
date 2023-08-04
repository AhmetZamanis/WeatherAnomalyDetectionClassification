# Canadian weather data - TS anomaly detection prep
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers.scaler import Scaler
from darts.ad.scorers.pyod_scorer import PyODScorer
from darts.ad.detectors.quantile_detector import QuantileDetector
from X_HelperFunctionsAnom import score, detect, plot_series, plot_dist
from X_HelperFunctionsAnom import plot_anom3d, plot_detection, plot_tsne


# Concatenate time covariates to Ottawa temperature series
ts_ottawa = ts_ottawa.concatenate(ts_covars, axis = 1)


# Split train-test
test_start = pd.Timestamp("1980-01-01")
train_end = pd.Timestamp("1979-12-31")
ts_train = ts_ottawa.drop_after(test_start)
ts_test = ts_ottawa.drop_before(train_end)


# Create scaler
feature_range = (-1, 1)
scaler = Scaler(MinMaxScaler(feature_range = feature_range))


# Create quantile detector
q = 0.999
detector = QuantileDetector(high_quantile = q)

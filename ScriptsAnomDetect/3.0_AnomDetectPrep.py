# Canadian weather data - TS anomaly detection prep
# Data source: https://openml.org/search?type=data&status=active&id=43843

exec(open("./ScriptsAnomDetect/1_DataPrep.py").read())


from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers.scaler import Scaler
from darts.ad.scorers.pyod_scorer import PyODScorer
from darts.ad.detectors.quantile_detector import QuantileDetector
from X_HelperFunctionsAnom import score, detect


# Concatenate time covariates to Ottawa temperature series
ts_ottawa = ts_ottawa.concatenate(ts_covars, axis = 1)


# Split train-test: Before vs. after 1980
ts_train = ts_ottawa.drop_after(pd.Timestamp("1980-01-01"))
ts_test = ts_ottawa.drop_before(pd.Timestamp("1979-12-31"))


# Create scaler
scaler = Scaler(MinMaxScaler(feature_range = (-1, 1)))


# Create quantile detector
detector = QuantileDetector(high_quantile = 0.99)

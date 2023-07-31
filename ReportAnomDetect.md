Time series anomaly detection - Canadian weather data
================
Ahmet Zamanis

- <a href="#introduction" id="toc-introduction">Introduction</a>
- <a href="#data-preparation" id="toc-data-preparation">Data
  preparation</a>
- <a href="#anomaly-detection" id="toc-anomaly-detection">Anomaly
  detection</a>
  - <a href="#k-means-clustering" id="toc-k-means-clustering">K-means
    clustering</a>
  - <a href="#gaussian-mixture-models-gmm"
    id="toc-gaussian-mixture-models-gmm">Gaussian mixture models (GMM)</a>
  - <a href="#ecod" id="toc-ecod">ECOD</a>
  - <a href="#principal-components-analysis-pca"
    id="toc-principal-components-analysis-pca">Principal components analysis
    (PCA)</a>
  - <a href="#isolation-forest" id="toc-isolation-forest">Isolation
    forest</a>
  - <a href="#autoencoder-with-pytorch-lightning"
    id="toc-autoencoder-with-pytorch-lightning">Autoencoder with PyTorch
    Lightning</a>
- <a href="#conclusion" id="toc-conclusion">Conclusion</a>

## Introduction

Introduce anomaly detection, packages, report structure

Introduce data & source

<details>
<summary>Show imports</summary>

``` python
# Data handling
import pandas as pd
import numpy as np
import arff # Installed as liac-arff
import warnings

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import IFrame

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller

# Darts
from darts.timeseries import TimeSeries
from darts.ad.scorers.kmeans_scorer import KMeansScorer
from darts.ad.scorers.pyod_scorer import PyODScorer
from darts.ad.detectors.quantile_detector import QuantileDetector

# PyOD anomaly scorers
from pyod.models.gmm import GMM
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.iforest import IForest

# Torch and Lightning
import torch
import lightning as L
from X_LightningClassesAnom import TrainDataset, TestDataset, AutoEncoder

# Hyperparameter tuning
import optuna

# Dimensionality reduction
from sklearn.manifold import TSNE

# Helper functions
from X_HelperFunctionsAnom import score, detect, plot_series, plot_dist, plot_anom3d, plot_detection, validate_nn
```

</details>
<details>
<summary>Show settings</summary>

``` python
# Set print options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

# Set plotting options
plt.rcParams['figure.dpi'] = 150
plt.rcParams["figure.autolayout"] = True
sns.set_style("darkgrid")
px_width = 800
px_height = 800

# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
L.seed_everything(1923, workers = True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
```

</details>

## Data preparation

``` python
# Load raw data from .arff
raw_data = arff.load(open("./InputData/canadian_climate.arff", "r"))

# Convert to Pandas dataframe & view
df = pd.DataFrame(
  raw_data["data"], columns = [x[0] for x in raw_data["attributes"]])
print(df.iloc[:,0:4])
```

                     LOCAL_DATE  MEAN_TEMPERATURE_CALGARY  \
    0      01-Jan-1940 00:00:00                  -11.4000   
    1      02-Jan-1940 00:00:00                  -12.0000   
    2      03-Jan-1940 00:00:00                  -12.0000   
    3      04-Jan-1940 00:00:00                  -11.4000   
    4      05-Jan-1940 00:00:00                  -13.1000   
    ...                     ...                       ...   
    29216  28-Dec-2019 00:00:00                   -7.7000   
    29217  29-Dec-2019 00:00:00                   -3.3000   
    29218  30-Dec-2019 00:00:00                   -1.6000   
    29219  31-Dec-2019 00:00:00                    4.3000   
    29220  01-Jan-2020 00:00:00                   -0.3000   

           TOTAL_PRECIPITATION_CALGARY  MEAN_TEMPERATURE_EDMONTON  
    0                           0.5000                        NaN  
    1                           0.5000                        NaN  
    2                           1.0000                        NaN  
    3                           0.8000                        NaN  
    4                           0.5000                        NaN  
    ...                            ...                        ...  
    29216                       0.0000                   -10.4000  
    29217                       0.0000                    -8.6000  
    29218                       0.0000                   -10.3000  
    29219                       0.0000                    -2.6000  
    29220                       0.0000                    -4.0000  

    [29221 rows x 4 columns]

``` python
# Convert LOCAL_DATE to datetime
df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])

# Add cyclic terms for month, week of year and day of year
df["month_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.month / 12)
df["week_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.isocalendar().week / 53)
df["week_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.isocalendar().week / 53)
df["day_sin"] = np.sin(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)
df["day_cos"] = np.cos(2 * np.pi * df["LOCAL_DATE"].dt.dayofyear / 366)
```

``` python
# Retrieve the weather data for Ottawa (1940 start) as Darts TimeSeries
ts_ottawa = TimeSeries.from_dataframe(
  df,
  time_col = "LOCAL_DATE",
  value_cols = ["MEAN_TEMPERATURE_OTTAWA", "TOTAL_PRECIPITATION_OTTAWA"],
  fill_missing_dates = True
)
ts_ottawa

# Interpolate missing values
na_filler = MissingValuesFiller()
ts_ottawa = na_filler.transform(ts_ottawa)
```

``` python
# Retrieve date covariates as Darts TS
ts_covars = TimeSeries.from_dataframe(
  df,
  time_col = "LOCAL_DATE",
  value_cols = ['month_sin', 'month_cos', 'week_sin', 'week_cos', 'day_sin', 
  'day_cos'],
  fill_missing_dates = True
)
print(ts_covars)
```

    <TimeSeries (DataArray) (LOCAL_DATE: 29221, component: 6, sample: 1)>
    array([[[ 0.5   ],
            [ 0.866 ],
            [ 0.1183],
            [ 0.993 ],
            [ 0.0172],
            [ 0.9999]],

           [[ 0.5   ],
            [ 0.866 ],
            [ 0.1183],
            [ 0.993 ],
            [ 0.0343],
            [ 0.9994]],

           [[ 0.5   ],
            [ 0.866 ],
            [ 0.1183],
            [ 0.993 ],
            [ 0.0515],
            [ 0.9987]],
    ...
           [[-0.    ],
            [ 1.    ],
            [ 0.1183],
            [ 0.993 ],
            [-0.0343],
            [ 0.9994]],

           [[-0.    ],
            [ 1.    ],
            [ 0.1183],
            [ 0.993 ],
            [-0.0172],
            [ 0.9999]],

           [[ 0.5   ],
            [ 0.866 ],
            [ 0.1183],
            [ 0.993 ],
            [ 0.0172],
            [ 0.9999]]])
    Coordinates:
      * LOCAL_DATE  (LOCAL_DATE) datetime64[ns] 1940-01-01 1940-01-02 ... 2020-01-01
      * component   (component) object 'month_sin' 'month_cos' ... 'day_cos'
    Dimensions without coordinates: sample
    Attributes:
        static_covariates:  None
        hierarchy:          None

``` python
# Concatenate time covariates to Ottawa weather series
ts_ottawa = ts_ottawa.concatenate(ts_covars, axis = 1)

# Split train-test data
test_start = pd.Timestamp("1980-01-01")
train_end = pd.Timestamp("1979-12-31")
ts_train = ts_ottawa.drop_after(test_start)
ts_test = ts_ottawa.drop_before(train_end)
```

## Anomaly detection

``` python
# Create Darts wrapper for MinMaxScaler
feature_range = (-1, 1) # The value range of cyclical encoded features
scaler = Scaler(MinMaxScaler(feature_range = feature_range))

# Create quantile detector
q = 0.99 # The detectpr will flag scores above this quantile as anomalies
detector = QuantileDetector(high_quantile = q)
```

### K-means clustering

``` python
# Create K-means scorer
scorer_kmeans = KMeansScorer(
  window = 1, # Score each time step by itself
  k = 12, # Number of K-means clusters
  random_state = 1923)

# Perform anomaly scoring
scores_train_kmeans, scores_test_kmeans, scores_kmeans = score(
  ts_train, ts_test, scorer_kmeans, scaler)
print(scores_kmeans)
```

    <TimeSeries (DataArray) (LOCAL_DATE: 29221, component: 1, sample: 1)>
    array([[[0.3829]],

           [[0.3652]],

           [[0.3068]],

           ...,

           [[0.5591]],

           [[0.5418]],

           [[0.4808]]])
    Coordinates:
      * LOCAL_DATE  (LOCAL_DATE) datetime64[ns] 1940-01-01 1940-01-02 ... 2020-01-01
      * component   (component) object '0'
    Dimensions without coordinates: sample
    Attributes:
        static_covariates:  None
        hierarchy:          None

``` python
# Perform anomaly detection
anoms_train_kmeans, anoms_test_kmeans, anoms_kmeans = detect(
  scores_train_kmeans, scores_test_kmeans, detector)
print(anoms_kmeans)
```

    <TimeSeries (DataArray) (LOCAL_DATE: 29221, component: 1, sample: 1)>
    array([[[0.]],

           [[0.]],

           [[0.]],

           ...,

           [[0.]],

           [[0.]],

           [[0.]]])
    Coordinates:
      * LOCAL_DATE  (LOCAL_DATE) datetime64[ns] 1940-01-01 1940-01-02 ... 2020-01-01
      * component   (component) object '0'
    Dimensions without coordinates: sample
    Attributes:
        static_covariates:  None
        hierarchy:          None

``` python
# Plot anomaly scores & original series
plot_series(
  "K-means scorer", ts_train, ts_test, scores_train_kmeans, scores_test_kmeans)
```

![](ReportAnomDetect_files/figure-commonmark/cell-12-output-1.png)

``` python
# Detections plot
plot_detection("K-means scores", q, ts_ottawa, scores_kmeans, anoms_kmeans)
```

![](ReportAnomDetect_files/figure-commonmark/cell-13-output-1.png)

``` python
# Plot distributions of anomaly scores
plot_dist("K-means scorer", scores_train_kmeans, scores_test_kmeans)
```

![](ReportAnomDetect_files/figure-commonmark/cell-14-output-1.png)

``` python
# 3D anomalies plot
plot_anom3d(
  "K-means scorer", ts_ottawa, anoms_kmeans, px_width, px_height, html = True)
```

<details>
<summary>Show code for 3D clustering plot</summary>

``` python
# Clustering plot
train_labels = scorer_kmeans.model.labels_.astype(str)
fig = px.scatter_3d(
  x = ts_train['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_train['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_train.time_index.month,
  color = train_labels,
  title = "K-Means clustering plot, train set",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Clusters"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/ClustKMeans.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/ClustKMeans.html", width=px_width, height=px_height)
```

</details>

        <iframe
            width="800"
            height="800"
            src="./HtmlPlots/ClustKMeans.html"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        

### Gaussian mixture models (GMM)

``` python
# Create PyOD GMM model
gmm = GMM(
  n_components = 4, # N. of Gaussian mixture components
  n_init = 10, # N. of initializations for expectation maximization
  contamination = 0.01, # % of expected anomalies in the dataset
  random_state = 1923)
  
# Create Darts GMM scorer  
scorer_gmm = PyODScorer(model = gmm, window = 1)
```

<details>
<summary>Show code for GMM anomaly detection</summary>

``` python
# Perform anomaly scoring
scores_train_gmm, scores_test_gmm, scores_gmm = score(
  ts_train, ts_test, scorer_gmm, scaler)

# Perform anomaly detection
anoms_train_gmm, anoms_test_gmm, anoms_gmm = detect(
  scores_train_gmm, scores_test_gmm, detector)

# Plot scores & original series
plot_series("GMM scorer", ts_train, ts_test, scores_train_gmm, scores_test_gmm)

# Detections plot
plot_detection("GMM scores", q, ts_ottawa, scores_gmm, anoms_gmm)

# Plot distributions of anomaly scores
plot_dist("GMM scorer", scores_train_gmm, scores_test_gmm)

# 3D anomalies plot
plot_anom3d("GMM scorer", ts_ottawa, anoms_gmm, px_width, px_height, html = True)

# 3D cluster plot
labels = scorer_gmm.model.detector_.predict(
  scaler.transform(ts_ottawa).values()).astype(str)
fig = px.scatter_3d(
  x = ts_ottawa['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts_ottawa['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts_ottawa.time_index.month,
  color = labels,
  title = "GMM clustering plot",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Clusters"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/ClustGMM.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/ClustGMM.html", width=px_width, height=px_height)
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-18-output-1.png)

![](ReportAnomDetect_files/figure-commonmark/cell-18-output-2.png)

![](ReportAnomDetect_files/figure-commonmark/cell-18-output-3.png)

        <iframe
            width="800"
            height="800"
            src="./HtmlPlots/ClustGMM.html"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        

### ECOD

``` python
# Create ECOD scorer
ecod = ECOD(contamination = 0.01)
scorer_ecod = PyODScorer(model = ecod, window = 1)
```

<details>
<summary>Show code for ECOD anomaly detection</summary>

``` python
# Perform anomaly scoring
scores_train_ecod, scores_test_ecod, scores_ecod = score(
  ts_train, ts_test, scorer_ecod)

# Perform anomaly detection
anoms_train_ecod, anoms_test_ecod, anoms_ecod = detect(
  scores_train_ecod, scores_test_ecod, detector)

# Plot scores & original series
plot_series("ECOD scorer", ts_train, ts_test, scores_train_ecod, scores_test_ecod)

# Detections plot
plot_detection("ECOD scores", q, ts_ottawa, scores_ecod, anoms_ecod)

# Plot distributions of anomaly scores
plot_dist("ECOD scorer", scores_train_ecod, scores_test_ecod)

# 3D anomalies plot
plot_anom3d("ECOD scorer", ts_ottawa, anoms_ecod, px_width, px_height, html = True)
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-20-output-1.png)

![](ReportAnomDetect_files/figure-commonmark/cell-20-output-2.png)

![](ReportAnomDetect_files/figure-commonmark/cell-20-output-3.png)

``` python
# Special to scorer: Explain a single training point's outlier scoring
idx_max_precip = np.argmax(
  ts_train['TOTAL_PRECIPITATION_OTTAWA'].univariate_values())
scorer_ecod.model.explain_outlier(
  ind = idx_max_precip, # Index of point to explain
  columns = [1, 2, 3, 4], # Dimensions to explain
  feature_names = ["MeanTemp", "TotalPrecip", "MonthSin", "MonthCos"])
plt.close("all")
```

![](ReportAnomDetect_files/figure-commonmark/cell-21-output-1.png)

### Principal components analysis (PCA)

``` python
# Create PCA scorer
pca = PCA(contamination = 0.01, standardization = True, random_state = 1923)
scorer_pca = PyODScorer(model = pca, window = 1)
```

<details>
<summary>Show code for PCA anomaly detection</summary>

``` python
# Perform anomaly scoring
scores_train_pca, scores_test_pca, scores_pca = score(
  ts_train, ts_test, scorer_pca)

# Perform anomaly detection
anoms_train_pca, anoms_test_pca, anoms_pca = detect(
  scores_train_pca, scores_test_pca, detector)

# Plot scores & original series
plot_series("PCA scorer", ts_train, ts_test, scores_train_pca, scores_test_pca)

# Detections plot
plot_detection("PCA scores", q, ts_ottawa, scores_pca, anoms_pca)

# Plot distributions of anomaly scores
plot_dist("PCA scorer", scores_train_pca, scores_test_pca)

# 3D anomalies plot
plot_anom3d("PCA scorer", ts_ottawa, anoms_pca, px_width, px_height, html = True)
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-23-output-1.png)

![](ReportAnomDetect_files/figure-commonmark/cell-23-output-2.png)

![](ReportAnomDetect_files/figure-commonmark/cell-23-output-3.png)

#### PC & T-SNE plots

``` python
# Variances explained by each component: First 3 PCs explain almost 99%
pc_variances = scorer_pca.model.explained_variance_ratio_ * 100
pc_variances = [round(x, 2) for x in pc_variances]
pc_variances
```

    [48.22, 37.31, 12.49, 1.56, 0.19, 0.19, 0.02, 0.01]

<details>
<summary>Show code to generate PC loadings heatplot</summary>

``` python
# Heatplot of PC loadings, X = PCs, Y = Features's contribution to the PCs
pc_loadings = pd.DataFrame(
  scorer_pca.model.components_.T,
  columns = pc_variances,
  index = ts_ottawa.components)
_ = sns.heatmap(pc_loadings, cmap = "PiYG")
_ = plt.title("PC loadings")
_ = plt.xlabel("% variances explained by PCs")
_ = plt.ylabel("Features")
plt.show()
plt.close("all")
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-25-output-1.png)

<details>
<summary>Show code to generate 3D PC plots</summary>

``` python
# Transform data into PC values
pcs = scorer_pca.model.detector_.transform(
  scorer_pca.model.scaler_.transform(ts_ottawa.values())
  )

# Principal components plot, first 3 PCs  
fig = px.scatter_3d(
  x = pcs[:, 0],
  y = pcs[:, 1],
  z = pcs[:, 2],
  color = anoms_pca.univariate_values().astype(str),
  title = "PCA components plot, first 3 PCs",
  labels = {
    "x": "PC1",
    "y": "PC2",
    "z": "PC3",
    "color": "Anomaly labels"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/PCS1.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/PCS1.html", width=px_width, height=px_height)


# Principal components plot, PC3-4-5
fig = px.scatter_3d(
  x = pcs[:, 4],
  y = pcs[:, 3],
  z = pcs[:, 2],
  color = anoms_pca.univariate_values().astype(str),
  title = "PCA components plot, middle 3 PCs",
  labels = {
    "x": "PC5",
    "y": "PC4",
    "z": "PC3",
    "color": "Anomaly labels"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/PCS2.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/PCS2.html", width=px_width, height=px_height)


# # Principal components plot, last 3 PCs
# fig = px.scatter_3d(
#   x = pcs[:, -1],
#   y = pcs[:, -2],
#   z = pcs[:, -3],
#   color = anoms_pca.univariate_values().astype(str),
#   title = "PCA components plot, last 3 PCs",
#   labels = {
#     "x": "PC6",
#     "y": "PC7",
#     "z": "PC8",
#     "color": "Anomaly labels"},
#     width = px_width,
#     height = px_height
# )
# fig.write_html("./HtmlPlot/plot.html", include_plotlyjs = "cdn")
# IFrame(src="./HtmlPlot/plot.html", width=800, height=600)
```

</details>

        <iframe
            width="800"
            height="800"
            src="./HtmlPlots/PCS2.html"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
<details>
<summary>Show code to generate 3D T-SNE plot</summary>

``` python
# Standardize & normalize PCs
std_scaler = StandardScaler()
pcs_scaled = std_scaler.fit_transform(pcs)

# Apply T-SNE to PCs
tsne = TSNE(n_components = 3)
z_pca = tsne.fit_transform(pcs_scaled)

# T-SNE dimensions plot
fig = px.scatter_3d(
  x = z_pca[:, 0],
  y = z_pca[:, 1],
  z = z_pca[:, 2],
  color = anoms_pca.univariate_values().astype(str),
  title = "PCA components plot, 3D T-SNE reduction",
  labels = {
    "x": "Dim1",
    "y": "Dim2",
    "z": "Dim3",
    "color": "Anomaly labels"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/TSNE1.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/TSNE1.html", width=px_width, height=px_height)
```

</details>

        <iframe
            width="800"
            height="800"
            src="./HtmlPlots/TSNE1.html"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        

### Isolation forest

``` python
# Create IForest scorer
iforest = IForest(
  n_estimators= 500, # N. of trees in forest
  contamination = 0.01,
  random_state = 1923)
scorer_iforest = PyODScorer(model = iforest, window = 1)
```

<details>
<summary>Show code for IForest anomaly detection</summary>

``` python
# Perform anomaly scoring
scores_train_iforest, scores_test_iforest, scores_iforest = score(
  ts_train, ts_test, scorer_iforest)

# Perform anomaly detection
anoms_train_iforest, anoms_test_iforest, anoms_iforest = detect(
  scores_train_iforest, scores_test_iforest, detector)

# Plot scores & original series
plot_series(
  "IForest scorer", ts_train, ts_test, scores_train_iforest, scores_test_iforest)

# Detections plot
plot_detection("IForest scores", q, ts_ottawa, scores_iforest, anoms_iforest)

# Plot distributions of anomaly scores
plot_dist("IForest scorer", scores_train_iforest, scores_test_iforest)

# 3D anomalies plot
plot_anom3d(
  "IForest scorer", ts_ottawa, anoms_iforest, px_width, px_height, html = True)
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-29-output-1.png)

![](ReportAnomDetect_files/figure-commonmark/cell-29-output-2.png)

![](ReportAnomDetect_files/figure-commonmark/cell-29-output-3.png)

<details>
<summary>Show code to plot IForest feature importances</summary>

``` python
# Feature importances (can be misleading for high cardinality features, e.g. day
# and week features)
feat_imp = pd.DataFrame({
  "Feature importance": scorer_iforest.model.feature_importances_,
  "Feature": ts_ottawa.components
}).sort_values("Feature importance", ascending = False)
_ = sns.barplot(data = feat_imp, x = "Feature importance", y = "Feature")
plt.show()
plt.close("all")
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-30-output-1.png)

### Autoencoder with PyTorch Lightning

#### Hyperparameter tuning

<details>
<summary>Show code to tune Autoencoder hyperparameters</summary>

``` python
# Split train - val data
val_start = pd.Timestamp("1970-01-01")
train_end = pd.Timestamp("1969-12-31")
ts_tr = ts_train.drop_after(val_start)
ts_val = ts_train.drop_before(train_end)

 # Perform preprocessing for train - validation split
x_tr = std_scaler.fit_transform(ts_tr.values())
x_val = std_scaler.transform(ts_val.values())

# Load data into TrainDataset
train_data = TrainDataset(x_tr)
val_data = TrainDataset(x_val)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 128, num_workers = 0, shuffle = True)
val_loader = torch.utils.data.DataLoader(
      val_data, batch_size = len(ts_val), num_workers = 0, shuffle = False)


# Define Optuna objective
def objective_nn(trial, train_loader, val_loader):

  # Define parameter ranges to tune over & suggest param set for trial
  hidden_size = trial.suggest_int("hidden_size", 2, 6, step = 2)
  latent_size = trial.suggest_int("latent_size", 1, (hidden_size - 1))
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)
  dropout = trial.suggest_float("dropout", 1e-4, 0.2)

  # Create hyperparameters dict
  hyperparams_dict = {
      "input_size": ts_train.values().shape[1],
      "hidden_size": hidden_size,
      "latent_size": latent_size,
      "learning_rate": learning_rate,
      "dropout": dropout}

  # Validate hyperparameter set
  score, epoch = validate_nn(hyperparams_dict, train_loader, val_loader, trial)

  # Report best n. of epochs
  trial.set_user_attr("n_epochs", epoch)

  return score


# Create study
study_nn = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_nn",
  direction = "minimize"
)

# Instantiate objective
obj = lambda trial: objective_nn(trial, train_loader, val_loader)

# Optimize study
study_nn.optimize(obj, n_trials = 500, show_progress_bar = True)

# Retrieve and export trials
trials_nn = study_nn.trials_dataframe().sort_values("value", ascending = True)
trials_nn.to_csv("./OutputData/trials_nnX.csv", index = False)
```

</details>

#### Anomaly detection

<details>
<summary>Show code to train & predict with Autoencoder model</summary>

``` python
# Import best trial
best_trial_nn = pd.read_csv("./OutputData/trials_nn1.csv").iloc[0,]

# Retrieve best hyperparameters
hyperparams_dict = {
      "input_size": ts_train.values().shape[1],
      "hidden_size": best_trial_nn["params_hidden_size"],
      "latent_size": best_trial_nn["params_latent_size"],
      "learning_rate": best_trial_nn["params_learning_rate"],
      "dropout": best_trial_nn["params_dropout"]}


# Perform preprocessing
x_train = std_scaler.fit_transform(ts_train.values())
x_test = std_scaler.transform(ts_test.values())

# Load data into TrainDataset & TestDataset
train_data = TrainDataset(x_train)
test_data = TestDataset(x_test)
train_score_data = TestDataset(x_train)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 128, num_workers = 0, shuffle = True)
test_loader = torch.utils.data.DataLoader(
      test_data, batch_size = len(ts_test), num_workers = 0, shuffle = False)
train_score_loader = torch.utils.data.DataLoader(
      train_score_data, batch_size = len(ts_train), num_workers = 0, shuffle = False)


# Create trainer
trainer = L.Trainer(
    max_epochs = int(best_trial_nn["user_attrs_n_epochs"]),
    accelerator = "gpu", devices = "auto", precision = "16-mixed",
    enable_model_summary = True,
    logger = True,
    enable_progress_bar = True,
    enable_checkpointing = True
    )

# Create & train model
model = AutoEncoder(hyperparams_dict = hyperparams_dict)
trainer.fit(model, train_loader)

# Perform reconstructions of training & testing data
preds_train = trainer.predict(
  model, train_score_loader)[0].cpu().numpy().astype(np.float64)
preds_test = trainer.predict(
  model, test_loader)[0].cpu().numpy().astype(np.float64)
```

</details>

    Training: 0it [00:00, ?it/s]

    Predicting: 0it [00:00, ?it/s]

    Predicting: 0it [00:00, ?it/s]

<details>
<summary>Show code to get reconstruction errors</summary>

``` python
# Perform anomaly scoring: Get mean reconstruction error for each datapoint

# Train set
scores_train = np.abs(x_train - preds_train) # Absolute errors of each dimensions
scores_train = np.mean(scores_train, axis = 1) # Mean absolute error over all dimensions
scores_train = pd.DataFrame(scores_train, index = ts_train.time_index) # Dataframe with corresponding dates
scores_train = TimeSeries.from_dataframe(scores_train) # Darts TS
scores_train = scores_train.with_columns_renamed("0", "Scores")

# Test set
scores_test = np.abs(x_test - preds_test)
scores_test = np.mean(scores_test, axis = 1) 
scores_test = pd.DataFrame(scores_test, index = ts_test.time_index)
scores_test = TimeSeries.from_dataframe(scores_test)
scores_test = scores_test.with_columns_renamed("0", "Scores")
scores = scores_train.append(scores_test)
```

</details>
<details>
<summary>Show code for Autoencoder anomaly detection</summary>

``` python
# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)

# Plot scores & original series
plot_series("Autoencoder scorer", ts_train, ts_test, scores_train, scores_test)

# Detections plot
plot_detection("Autoencoder scores", q, ts_ottawa, scores, anoms)

# Plot distributions of anomaly scores
plot_dist("Autoencoder scorer", scores_train, scores_test)

# 3D anomalies plot
plot_anom3d(
  "Autoencoder scorer", ts_ottawa, anoms, px_width, px_height, html = True)
```

</details>

![](ReportAnomDetect_files/figure-commonmark/cell-34-output-1.png)

![](ReportAnomDetect_files/figure-commonmark/cell-34-output-2.png)

![](ReportAnomDetect_files/figure-commonmark/cell-34-output-3.png)

#### Latent space plot with T-SNE

<details>
<summary>Show code to generate 3D T-SNE plot</summary>

``` python
# Define function to get latent space representations
def get_latent(model, dataloader):
  model.eval() # Put model into inference mode
  with torch.no_grad(): # Disable gradient calculation
    z = [model.forward(x) for _, x in enumerate(dataloader)]
    return z[0].cpu().numpy().astype(np.float64)

# Retrieve & concatenate latent space representations
z_train = get_latent(model, train_score_loader)
z_test = get_latent(model, test_loader)
z = np.concatenate((z_train, z_test), axis = 0)


# Apply T-SNE to latent space
z_scaled = std_scaler.fit_transform(z)
z_autoencoder = tsne.fit_transform(z_scaled)


# Latent space plot
fig = px.scatter_3d(
  x = z_autoencoder[:, 0],
  y = z_autoencoder[:, 1],
  z = z_autoencoder[:, 2],
  color = anoms.univariate_values().astype(str),
  title = "Autoencoder latent space plot, 3D T-SNE reduction",
  labels = {
    "x": "Dim1",
    "y": "Dim2",
    "z": "Dim3",
    "color": "Anomaly labels"},
    width = px_width,
    height = px_height
)
fig.write_html("./HtmlPlots/TSNE2.html", include_plotlyjs = "cdn")
IFrame(src="./HtmlPlots/TSNE2.html", width=px_width, height=px_height)
```

</details>

        <iframe
            width="800"
            height="800"
            src="./HtmlPlots/TSNE2.html"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        

## Conclusion

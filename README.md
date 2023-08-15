# WeatherAnomalyDetectionClassification
This repository holds the scripts and reports for a project on time series anomaly detection, time series classification & dynamic time warping, performed on a dataset of Canadian weather measurements. The data was sourced from [OpenML](https://openml.org/search?type=data&status=active&id=43843&sort=runs), shared by user Elif Ceren Gök.

## Time series anomaly detection
Multivariate time series anomaly detection using [PyOD](https://github.com/yzhao062/pyod) algorithms & the [Darts](https://github.com/unit8co/darts) package: K-means clustering, Gaussian Mixture Models, ECOD, Isolation Forest and an Autoencoder with PyTorch Lightning. Visualizing & comparing the results with multiple plots, including 3D interactive Plotly scatterplots. 
\
[Full report](https://ahmetzamanis.github.io/WeatherAnomalyDetectionClassification/)
\
[Scripts](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/tree/main/ScriptsAnomDetect), [Lightning classes](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/blob/main/X_LightningClassesAnom.py), [functions](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/blob/main/X_HelperFunctionsAnom.py) 

## Time series classification
Multivariate time series classification using [sktime](https://github.com/sktime/sktime) and [pyts](https://github.com/johannfaouzi/pyts): kNN with DTW distance, ROCKET & Arsenal, WEASELMUSE and a PyTorch Lightning convolutional neural network trained on image transformed data. Visualizing & comparing the performances of all algorithms.
\
[Full report](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/blob/main/ReportClassification.md)
\
[Scripts](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/tree/main/ScriptsClassification), [Lightning classes](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/blob/main/X_LightningClassesClassif.py), [functions](https://github.com/AhmetZamanis/WeatherAnomalyDetectionClassification/blob/main/X_HelperFunctionsClassif.py)

# Canadian weather data - TS anomaly detection with PCA
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.pca import PCA
from sklearn.preprocessing import StandardScaler


# Create PCA scorer
pca = PCA(contamination = 0.01, standardization = True, random_state = 1923)
scorer = PyODScorer(model = pca, window = 1)
scorer_name = "PCA scorer"


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
# No "mistaken" anomalies close to the centers of any month.


# Detections plot
plot_detection("PCA scores", q, ts_ottawa, scores, anoms)
# PCA seems to select solely hot & unusually rainy days as anomalies.


# Variances explained by each component: First 3 PCs explain almost 99%
pc_variances = scorer.model.explained_variance_ratio_ * 100
pc_variances = [round(x, 2) for x in pc_variances]

# Heatplot of PC loadings, X = PCs, Y = Features's contribution to the PCs
pc_loadings = pd.DataFrame(
  scorer.model.components_.T,
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


# Transform data into PC values
pc_train = scorer.model.detector_.transform(
  scorer.model.scaler_.transform(ts_train.values())
  )
pc_test = scorer.model.detector_.transform(
  scorer.model.scaler_.transform(ts_test.values())
)
pcs = np.concatenate((pc_train, pc_test), axis = 0)


# Principal components plot, first 3 PCs
fig = px.scatter_3d(
  x = pcs[:, 0],
  y = pcs[:, 1],
  z = pcs[:, 2],
  color = anoms.univariate_values().astype(str),
  title = "PCA components plot, first 3 PCs",
  labels = {
    "x": "PC1",
    "y": "PC2",
    "z": "PC3",
    "color": "Anomaly labels"}
)
fig.show()


# Principal components plot, PC3-4-5
fig = px.scatter_3d(
  x = pcs[:, 4],
  y = pcs[:, 3],
  z = pcs[:, 2],
  color = anoms.univariate_values().astype(str),
  title = "PCA components plot, middle 3 PCs",
  labels = {
    "x": "PC5",
    "y": "PC4",
    "z": "PC3",
    "color": "Anomaly labels"}
)
fig.show()


# Principal components plot, last 3 PCs
fig = px.scatter_3d(
  x = pcs[:, -1],
  y = pcs[:, -2],
  z = pcs[:, -3],
  color = anoms.univariate_values().astype(str),
  title = "PCA components plot, last 3 PCs",
  labels = {
    "x": "PC6",
    "y": "PC7",
    "z": "PC8",
    "color": "Anomaly labels"}
)
fig.show()
# It's clear PC3 is the most important dimension for separating anomalies. PC4
# is maybe a bit important too. Makes sense because PC3 is practically precipitation
# and PC4 is practically temperature.
# The plot of the first 3 PCs reflects the cyclical nature of the data.


# Standardize & normalize PCs
std_scaler = StandardScaler()
pcs_scaled = std_scaler.fit_transform(pcs)


# Apply T-SNE to PCs
# Perp 5: Literal blob, anoms at center
# Perp 15: Still blobby, but circular strings start to emerge in clusters, anoms at center
# Perp 30: Clusters even more distinct, some circular some string-like, anoms at center
# Pert 45: Not too different from 30
plot_tsne(pcs_scaled, anoms, px_width, px_height, perplexities = [15, 30, 45])


















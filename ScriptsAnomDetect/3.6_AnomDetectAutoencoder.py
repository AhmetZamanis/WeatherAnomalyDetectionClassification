# Canadian weather data - TS anomaly detection with Autoencoder
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())


from pyod.models.auto_encoder_torch_latent import AutoEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Create AutoEncoder scorer
autoencoder = AutoEncoder(
  hidden_neurons = [6, 4],
  dropout_rate = 0.05,
  preprocessing = True,
  contamination = 0.01)
scorer = PyODScorer(model = autoencoder, window = 1)
scorer_name = "AutoEncoder scorer"


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


# Detections plot
plot_detection("AutoEncoder scores", q, ts_ottawa, scores, anoms)


# Get latent space
z = scorer.model.latent_space(ts_ottawa.values())


# Apply T-SNE to latent space
scaler = StandardScaler()
z_scaled = scaler.fit_transform(z)
tsne = TSNE(n_components = 3)
z_tsne = tsne.fit_transform(z_scaled)
z_tsne.shape


# Latent space plot
fig = px.scatter_3d(
  x = z_tsne[:, 0],
  y = z_tsne[:, 1],
  z = z_tsne[:, 2],
  color = anoms.univariate_values().astype(str),
  title = "Autoencoder latent space plot (3-dimensional T-SNE)",
  labels = {
    "x": "D1",
    "y": "D2",
    "z": "D3",
    "color": "Anomaly labels"}
)
fig.show()




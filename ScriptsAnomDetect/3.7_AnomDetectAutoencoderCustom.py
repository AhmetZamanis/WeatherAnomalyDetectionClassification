# Canadian weather data - TS anomaly detection with custom Autoencoder
# Data source: https://openml.org/search?type=data&status=active&id=43843


# Source data prep script
exec(open("./ScriptsAnomDetect/3.0_AnomDetectPrep.py").read())

import torch
import lightning as L
import warnings
import optuna
from X_LightningClassesAnom import TrainDataset, TestDataset, AutoEncoder
from X_HelperFunctionsAnom import validate_nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
L.seed_everything(1923, workers = True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# Split train - val data
val_start = pd.Timestamp("1970-01-01")
train_end = pd.Timestamp("1969-12-31")
ts_tr = ts_train.drop_after(val_start)
ts_val = ts_train.drop_before(train_end)


 # Perform preprocessing for train - validation split
scaler = StandardScaler()
x_tr = scaler.fit_transform(ts_tr.values())
x_val = scaler.transform(ts_val.values())

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
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)
  lr_decay = trial.suggest_float("lr_decay", 0.9, 1)
  dropout = trial.suggest_float("dropout", 1e-4, 0.2)

  # Create hyperparameters dict
  hyperparams_dict = {
      "input_size": ts_train.values().shape[1],
      "learning_rate": learning_rate,
      "lr_decay": lr_decay,
      "dropout": dropout
    }

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
study_nn.optimize(
  obj,
  n_trials = 500,
  show_progress_bar = True)


# Retrieve and export trials
trials_nn = study_nn.trials_dataframe().sort_values("value", ascending = True)
trials_nn.to_csv("./OutputData/trials_nnX.csv", index = False)


# Import best trial
best_trial_nn = pd.read_csv("./OutputData/trials_nn2.csv").iloc[0,]


# Retrieve best hyperparameters
hyperparams_dict = {
      "input_size": ts_train.values().shape[1],
      "learning_rate": best_trial_nn["params_learning_rate"],
      "lr_decay": best_trial_nn["params_lr_decay"],
      "dropout": best_trial_nn["params_dropout"]
    }


# Perform preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(ts_train.values())
x_test = scaler.transform(ts_test.values())

# Load data into TrainDataset
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
    # max_epochs = hyperparams_dict["n_epochs"],
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


# # Load trained model
# ckpt_path = "./lightning_logs/version_0/checkpoints/epoch=60-step=7015.ckpt"
# model = AutoEncoder.load_from_checkpoint(ckpt_path)


# Perform anomaly scoring: Get mean reconstruction error for each datapoint
# DON'T PRINT SCORES TIMESERIES AFTER CREATION, IT HANGS PYTHON
scores_train = trainer.predict(model, train_score_loader)[0].cpu().numpy().astype(np.float64)
scores_train = np.abs(x_train - scores_train)
scores_train = np.mean(scores_train, axis = 1) 
scores_train = pd.DataFrame(scores_train, index = ts_train.time_index)
scores_train = TimeSeries.from_dataframe(scores_train)
scores_train = scores_train.with_columns_renamed("0", "Scores")

scores_test = trainer.predict(model, test_loader)[0].cpu().numpy().astype(np.float64)
scores_test = np.abs(x_test - scores_test)
scores_test = np.mean(scores_test, axis = 1) 
scores_test = pd.DataFrame(scores_test, index = ts_test.time_index)
scores_test = TimeSeries.from_dataframe(scores_test)
scores_test = scores_test.with_columns_renamed("0", "Scores")
scores = scores_train.append(scores_test)


# Perform anomaly detection
anoms_train, anoms_test, anoms = detect(scores_train, scores_test, detector)


# Plot scores & original series
scorer_name = "Autoencoder scorer"
plot_series(scorer_name, ts_train, ts_test, scores_train, scores_test)


# Plot distributions of anomaly scores
plot_dist(scorer_name, scores_train, scores_test)


# 3D anomalies plot
plot_anom3d(scorer_name, ts_ottawa, anoms, px_width, px_height)
# Only a few highest precip. days in summer & fall are labeled as anomalies,
# along with many "false" anomalies in spring & fall. Maybe the best performance
# with a higher threshold?


# Detections plot
plot_detection("Autoencoder scores", q, ts_ottawa, scores, anoms)
# The 2000> days with very high precip. are all scored very highly, but many of
# the highly scored days seem normal


# Get latent space representations
def get_latent(model, dataloader):
  model.eval()
  with torch.no_grad():
    z = [model.forward(x) for _, x in enumerate(dataloader)]
    return z[0].cpu().numpy().astype(np.float64)

z_train = get_latent(model, train_score_loader)
z_test = get_latent(model, test_loader)
z = np.concatenate((z_train, z_test), axis = 0)


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
# The plot seems to have unique string-like manifolds for each month / time period.
# The sub-manifolds are fewer in number & more joined compared to PCAs.
# Most anomalies are false anomalies so they don't have distinct separation from
# the closest manifolds. 


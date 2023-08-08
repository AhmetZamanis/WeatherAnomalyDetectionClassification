# Functions for TS anomaly detection scripts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from X_LightningClassesAnom import AutoEncoder, OptunaPruning
from sklearn.manifold import TSNE


def score(ts_train, ts_test, scorer, scaler = None):
  """
  Train scorer & score series. Perform scaling if necessary.
  """
  
  # Scale series if applicable
  if scaler:
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)
  
  # Train scorer
  _ = scorer.fit(ts_train)
  
  # Score series
  scores_train = scorer.score(ts_train)
  scores_test = scorer.score(ts_test)
  scores = scores_train.append(scores_test)
  
  return scores_train, scores_test, scores


def detect(scores_train, scores_test, detector):
  """
  Train detector & detect series.
  """
  
  # Train & detect
  anoms_train = detector.fit_detect(scores_train)
  anoms_test = detector.detect(scores_test)
  anoms = anoms_train.append(anoms_test)
  
  return anoms_train, anoms_test, anoms


def plot_series(scorer_name, ts_train, ts_test, scores_train, scores_test):
  """
  Plot anomaly scores, original series in train-test splits.
  """
  
  # Create figure
  fig, ax = plt.subplots(3, sharex = True)

  # Plot scores
  scores_train.plot(ax = ax[0])
  scores_test.plot(ax = ax[0])
  _ = ax[0].set_title("Anomaly scores, " +  scorer_name)

  # Plot MeanTemp
  ts_train['MEAN_TEMPERATURE_OTTAWA'].plot(ax = ax[1])
  ts_test['MEAN_TEMPERATURE_OTTAWA'].plot(ax = ax[1])
  _ = ax[1].set_title("Mean temperatures")

  # Plot TotalPrecip
  ts_train['TOTAL_PRECIPITATION_OTTAWA'].plot(label = "Train set", ax = ax[2])
  ts_test['TOTAL_PRECIPITATION_OTTAWA'].plot(label = "Test set", ax = ax[2])
  _ = ax[2].set_title("Total precipitation")

  plt.show()
  plt.close("all")


def plot_dist(scorer_name, scores_train, scores_test):
  """
  Plot distributions of anomaly scores for train-test sets.
  """
  
  # Get data
  df_train = scores_train.pd_dataframe().rename({"0": "Scores"}, axis = 1)
  df_test = scores_test.pd_dataframe().rename({"0": "Scores"}, axis = 1)
  df_train["Set"] = "Train"
  df_test["Set"] = "Test"
  df = pd.concat([df_train, df_test])
  
  _ = sns.kdeplot(data = df, x = "Scores", hue = "Set")
  _ = plt.title("Distributions of " + scorer_name + " anomaly scores")
  bottom, top = plt.ylim()
  _ = plt.ylim(-top * 0.05, top)
  plt.show()
  plt.close("all")


def plot_anom3d(scorer_name, ts, anoms, px_width, px_height):
  """
  Plot 3D scatterplot of anomalies.
  """
  
  fig = px.scatter_3d(
  x = ts['MEAN_TEMPERATURE_OTTAWA'].univariate_values(),
  y = ts['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(),
  z = ts.time_index.month,
  color = anoms.univariate_values().astype(str),
  title = scorer_name + " anomalies plot",
  labels = {
    "x": "Mean temperature",
    "y": "Total precipitation",
    "z": "Month",
    "color": "Anomaly labels"},
    width = px_width,
    height = px_height
  )
  fig.show()


def plot_detection(scores_name, quantile, ts, scores, anoms):
  """
  Plot "detection lineplots" of anomaly scores & original series, with different 
  colors for anomalous & non-anomalous time steps.
  """
  
  q_str = str(quantile)
  
  # Retrieve dates, variables and anomaly labels in dataframes, separately for
# positive and negative observations
  df_positive = pd.DataFrame({
    "Date": ts.time_index,
    scores_name: np.where(
      anoms.univariate_values() == 1, 
      scores.univariate_values(), np.nan),
    "Mean temperature": np.where(
      anoms.univariate_values() == 1, 
      ts['MEAN_TEMPERATURE_OTTAWA'].univariate_values(), np.nan),
    "Total precipitation": np.where(
      anoms.univariate_values() == 1, 
      ts['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(), np.nan)
    }
  )

  df_negative = pd.DataFrame({
    "Date": ts.time_index,
    scores_name: np.where(
      anoms.univariate_values() == 0, 
      scores.univariate_values(), np.nan),
    "Mean temperature": np.where(
      anoms.univariate_values() == 0, 
      ts['MEAN_TEMPERATURE_OTTAWA'].univariate_values(), np.nan),
    "Total precipitation": np.where(
      anoms.univariate_values() == 0, 
      ts['TOTAL_PRECIPITATION_OTTAWA'].univariate_values(), np.nan)
    }
  )


  # Plot original series, colored by anomalous & non-anomalous
  fig, ax = plt.subplots(3, sharex = True)
  _ = fig.suptitle("Anomaly detections with " + q_str + "th quantile " + scores_name + "\nBlue = Anomalous days")

  _ = sns.lineplot(data = df_negative,  x = "Date",  y = scores_name, ax = ax[0])
  _ = sns.lineplot(data = df_positive,  x = "Date",  y = scores_name, ax = ax[0])

  _ = sns.lineplot(data = df_negative,  x = "Date",  y = "Mean temperature", ax = ax[1])
  _ = sns.lineplot(data = df_positive,  x = "Date",  y = "Mean temperature", ax = ax[1])

  _ = sns.lineplot(data = df_negative, x = "Date", y = "Total precipitation", ax = ax[2])
  _ = sns.lineplot(data = df_positive, x = "Date", y = "Total precipitation", ax = ax[2])

  plt.show()
  plt.close("all")


def plot_tsne(z, anoms, px_width, px_height, perplexities, n_components = 3, n_iter = 5000, random_state = 1923):
  """
  Performs T-SNE reductions with the given perplexity values, 3D plots the results
  colored by anomaly labels.
  """
  
  for p in perplexities:
    
    # Perform T-SNE
    tsne = TSNE(
      n_components = n_components,
      perplexity = p,
      n_iter = n_iter,
      n_iter_without_progress = int(n_iter / 10),
      random_state = random_state)
    z = tsne.fit_transform(z)
    
    # Plot the results
    fig = px.scatter_3d(
      x = z[:, 0],
      y = z[:, 1],
      z = z[:, 2],
      color = anoms.univariate_values().astype(str),
      title = "T-SNE plot, perplexity=" + str(p),
      labels = {
        "x": "D1",
        "y": "D2",
        "z": "D3",
        "color": "Anomaly labels"},
      width = px_width,
      height = px_height
      )
    fig.show()


def validate_nn(hyperparams_dict, train_loader, val_loader, trial, tol = 1e-4):
  """
  Validate a set of Torch AutoEncoder parameters & report to Optuna.
  """
  
  # Create callbacks list
  callbacks = []
    
  # Create early stop callback
  callback_earlystop = EarlyStopping(
      monitor = "val_loss", mode = "min", min_delta = tol, patience = 10)
  callbacks.append(callback_earlystop)
    
  # Create Optuna pruner callback
  callback_pruner = OptunaPruning(trial, monitor = "val_loss")
  callbacks.append(callback_pruner)
    
  # Create trainer
  trainer = L.Trainer(
    max_epochs = 500,
    accelerator = "gpu", devices = "auto", precision = "16-mixed", 
    callbacks = callbacks,
    enable_model_summary = False, 
    logger = True,
    enable_progress_bar = False, # Disable prog. bar, checkpoints for Optuna trials
    enable_checkpointing = False
    )
  
  # Create & train model
  model = AutoEncoder(hyperparams_dict = hyperparams_dict)
  trainer.fit(model, train_loader, val_loader)
  
  # Retrieve best val score and n. of epochs
  score = callback_earlystop.best_score.cpu().numpy()
  epoch = trainer.current_epoch - callback_earlystop.wait_count # Starts from 1
  
  # Return score & epoch
  return score, epoch
  
 

# Functions for TS classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightning as L

from lightning.pytorch.callbacks import EarlyStopping
from X_LightningClassesClassif import CNN, OptunaPruning
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss


def plot_confusion(y_true, y_pred, labels, title_str):
  """
  Plots a confusion matrix heatplot given the multiclass predictions and actual
  classes.
  """
  matrix = confusion_matrix(y_true, y_pred, labels = labels)
  _ = sns.heatmap(
      matrix, xticklabels = labels, yticklabels = labels, cmap = "Reds", 
      annot = True, fmt = "g", square = True, cbar = False, linecolor = "black", 
      linewidths = 0.5)
  _ = plt.xlabel("Predicted classes")
  _ = plt.ylabel("True classes")
  _ = plt.title("Confusion matrix, " + title_str)
  plt.show()
  plt.close("all")
    

def get_images(x_train, x_test, trafo):
  """
  Takes in training & inference data as 3Darrays of shape 
  (n_seq, n_dims, seq_length).
  Returns recurrence plot multichannel image data of shape 
  (n_seq, n_channels, seq_length, seq_length), with each channel scaled 0-1.
  """
  
  # Retrieve number of dimensions
  n_dims = x_train.shape[1]
  
  # Retrieve each dimension in a list
  dims_train = [x_train[:, i, :] for i in range(0, n_dims)]
  dims_test = [x_test[:, i, :] for i in range(0, n_dims)]
  
  # Fit & transform training data for each dimension
  fitted_trafos = []
  fitted_mins = []
  fitted_maxi = []
  channels_train = []
  channels_test = []
  for dim in dims_train:
    
    # Copy base transformer
    fitted_trafo = trafo
    
    # Fit & transform dimension
    transformed = fitted_trafo.fit_transform(dim)
    
    # Save fitted transformer
    fitted_trafos.append(fitted_trafo)
    
    # Retrieve & save min, max of the dimension matrix
    dim_min = np.min(transformed)
    dim_max = np.max(transformed)
    fitted_mins.append(dim_min)
    fitted_maxi.append(dim_max)
    
    # Scale values between 0 and 1, append to list
    transformed_scaled = (transformed - dim_min) / (dim_max - dim_min)
    channels_train.append(transformed_scaled)
    
  # Transform testing data for each dimension
  for i, dim in enumerate(dims_test):
    
    # Transform & scale dimension, append to list
    transformed = fitted_trafos[i].transform(dim)
    transformed_scaled = (transformed - fitted_mins[i]) / (fitted_maxi[i] - fitted_mins[i])
    channels_test.append(transformed_scaled)
    
  # Stack the outputs
  output_train = np.stack(channels_train, axis = 1)
  output_test = np.stack(channels_test, axis = 1)

  return output_train, output_test


def scale_dims(x_train, x_test):
  """
  Performs dimension-wise 0-1 scaling given a 3Darray input of shape 
  (n_seq, n_dims, seq_length).
  """
  
  # Retrieve number of dimensions
  n_dims = x_train.shape[1]
  
  # Retrieve each dimension in a list
  dims_train = [x_train[:, i, :] for i in range(0, n_dims)]
  dims_test = [x_test[:, i, :] for i in range(0, n_dims)]
  
  # Fit & transform training data for each dimension
  fitted_mins = []
  fitted_maxi = []
  output_train = []
  output_test = []
  
  for dim in dims_train:
    
    # Retrieve & save min, max of the dimension matrix
    dim_min = np.min(dim)
    dim_max = np.max(dim)
    fitted_mins.append(dim_min)
    fitted_maxi.append(dim_max)
    
    # Scale values between 0 and 1, append to list
    transformed = (dim - dim_min) / (dim_max - dim_min)
    output_train.append(transformed)
    
  # Transform testing data for each dimension
  for i, dim in enumerate(dims_test):
    
    # Scale dimension, append to list
    transformed = (dim - fitted_mins[i]) / (fitted_maxi[i] - fitted_mins[i])
    output_test.append(transformed)
    
  # Stack the outputs
  output_train = np.stack(output_train, axis = 1)
  output_test = np.stack(output_test, axis = 1)

  return output_train, output_test


def plot_images(data, dim, dimname, seq1, seq2, seq_per_city):
  """
  Plots the recurrence plot for two sequences, all cities, one dimension.
  """
  
  # Create figure
  fig, ax = plt.subplots(3, 2, sharex = True, sharey = True)
  _ = plt.suptitle(
  "Recurrence plots of two 28-day periods in the data, dimension: " + 
  dimname)
  
  # Ottawa seq1
  _ = ax[0, 0].imshow(data[seq1][dim], cmap = "binary", origin = "lower")
  _ = ax[0, 0].set_title("Ottawa")
  
  # Toronto seq1
  _ = ax[1, 0].imshow(
    data[seq1 + seq_per_city][dim], cmap = "binary", origin = "lower")
  _ = ax[1, 0].set_title("Toronto")
  
  # Vancouver seq1
  _ = ax[2, 0].imshow(
    data[seq1 + (seq_per_city * 2)][dim], cmap = "binary", origin = "lower")
  _ = ax[2, 0].set_title("Vancouver")
  
  # Ottawa seq2
  _ = ax[0, 1].imshow(data[seq2][dim], cmap = "binary", origin = "lower")
  _ = ax[0, 1].set_title("Ottawa")
  
  # Toronto seq2
  _ = ax[1, 1].imshow(
    data[seq2 + seq_per_city][dim], cmap = "binary", origin = "lower")
  _ = ax[1, 1].set_title("Toronto")
  
  # Vancouver seq2
  _ = ax[2, 1].imshow(
    data[seq2 + (seq_per_city * 2)][dim], cmap = "binary", origin = "lower")
  _ = ax[2, 1].set_title("Vancouver")
  plt.show()
  plt.close("all")


def validate_cnn(hyperparams_dict, train_loader, val_loader, trial, tol = 1e-4):
  """
  Validate a set of Torch CNN parameters & report to Optuna.
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
  model = CNN(hyperparams_dict = hyperparams_dict)
  trainer.fit(model, train_loader, val_loader)
  
  # Retrieve best val score and n. of epochs
  score = callback_earlystop.best_score.cpu().numpy()
  epoch = trainer.current_epoch - callback_earlystop.wait_count # Starts from 1
  
  # Return score & epoch
  return score, epoch


def test_model(model, x_train, x_test, y_train, y_test, scale = False):
  """
  Takes in a sktime classifier, optionally performs scaling, tests model, returns
  accuracy, log loss and test set predictions.
  """
  
  # Scale the features
  x_train, x_test = scale_dims(x_train, x_test)

  # Fit on training data
  _ = model.fit(x_train, y_train)

  # Predict testing data
  preds = model.predict(x_test)
  probs = model.predict_proba(x_test)
  
  # Get class labels
  classes = np.unique(y_train)

  # Calculate accuracy
  acc = accuracy_score(y_test, preds)

  # Calculate log loss
  loss = log_loss(y_test, probs, labels = classes)
  
  return preds, probs, acc, loss

# Canadian weather data - TS classification with recurrence plots + CNN
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1.1_RecurrencePlot.py").read())


import torch
import lightning as L
import warnings
import optuna

from sklearn.preprocessing import OneHotEncoder


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
L.seed_everything(1923, workers = True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# Convert multiclass targets into binary matrices of (n_seq, n_classes)
encoder_onehot = OneHotEncoder(sparse_output = False)
y_train = encoder_onehot.fit_transform(y_train.reshape(-1, 1))
y_test = encoder_onehot.fit_transform(y_test.reshape(-1, 1))



# Load data into TrainDataset


# Create dataloaders


# Optuna tuning


# Export - import - retrieve best tune


# Load data into TrainDataset and TestDataset

# Create dataloaders

# Create trainer

# Create & train model

# Predict testing data


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(y_test, preds_cnn)

# Log loss
log_loss(y_test, probs_cnn, labels = classes)


# Plot confusion matrix
plot_confusion(y_test, preds_cnn, classes, "Recurrence plot + CNN classifier")

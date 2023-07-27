# Canadian weather data - TS classification with recurrence plots + CNN
# Data source: https://openml.org/search?type=data&status=active&id=43843

# Source data prep script
exec(open("./ScriptsClassification/1.1_RecurrencePlot.py").read())


import torch
import lightning as L
import warnings
import optuna

from sklearn.preprocessing import OneHotEncoder
from X_LightningClassesClassif import TrainDataset, TestDataset, CNN, OptunaPruning
from X_HelperFunctionsClassif import validate_cnn, plot_confusion


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
L.seed_everything(1923, workers = True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# Convert multiclass targets into binary matrices of (n_seq, n_classes)
encoder_onehot = OneHotEncoder(sparse_output = False)
y_train = encoder_onehot.fit_transform(y_train.reshape(-1, 1))
y_test = encoder_onehot.fit_transform(y_test.reshape(-1, 1))
y_tr = encoder_onehot.fit_transform(y_tr.reshape(-1, 1))
y_val = encoder_onehot.fit_transform(y_val.reshape(-1, 1))


# Load data into TrainDataset
tr_data = TrainDataset(x_tr, y_tr)
val_data = TrainDataset(x_val, y_val)


# Create dataloaders
tr_loader = torch.utils.data.DataLoader(
  train_data, batch_size = 128, num_workers = 0, shuffle = True)
val_loader = torch.utils.data.DataLoader(
  val_data, batch_size = len(val_data), num_workers = 0, shuffle = False)


# Define Optuna objective
def objective_cnn(trial, tr_loader, val_loader):
  
  # Define parameter ranges to tune over
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)
  lr_decay = trial.suggest_float("lr_decay", 0.9, 1)
  
  # Create hyperparameters dict
  hyperparams_dict = {
    "input_channels": x_tr.shape[1],
    "learning_rate": learning_rate,
    "lr_decay": lr_decay
  }
  
  # Validate hyperparameter set
  score, epoch = validate_cnn(hyperparams_dict, tr_loader, val_loader, trial)
  
  # Report best n. of epochs to study
  trial.set_user_attr("n_epochs", epoch)
  
  return score


# Create study
study_cnn = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_cnn",
  direction = "minimize"
)


# Instantiate objective
obj = lambda trial: objective_cnn(trial, tr_loader, val_loader)


# Optimize study
study_cnn.optimize(obj, n_trials = 500, show_progress_bar = True)


# Retrieve and export trials
trials_cnn = study_cnn.trials_dataframe().sort_values("value", ascending = True)
trials_cnn.to_csv("./OutputData/trials_cnn1.csv", index = False)


# Import best trial
best_trial_cnn = pd.read_csv("./OutputData/trials_cnn1.csv").iloc[0,]


# Retrieve best hyperparameters
hyperparams_dict = {
    "input_channels": x_tr.shape[1],
    "learning_rate": best_trial_cnn["params_learning_rate"],
    "lr_decay": best_trial_cnn["params_lr_decay"]
  }


# Load data into TrainDataset
train_data = TrainDataset(x_train, y_train)
test_data = TestDataset(x_test)


# Create dataloaders
train_loader = torch.utils.data.DataLoader(
  train_data, batch_size = 128, num_workers = 0, shuffle = True)
test_loader = torch.utils.data.DataLoader(
  test_data, batch_size = len(test_data), num_workers = 0, shuffle = False)


# Create trainer
trainer = L.Trainer(
  max_epochs = int(best_trial_cnn["user_attrs_n_epochs"]),
  accelerator = "gpu", devices = "auto", precision = "16-mixed",
  enable_model_summary = True,
  logger = True,
  enable_progress_bar = True,
  enable_checkpointing = True
)


# Create & train model
model = CNN(hyperparams_dict)
trainer.fit(model, train_loader)


# Predict testing data
probs_cnn = trainer.predict(model, test_loader)
probs_cnn = probs_cnn[0].cpu().numpy().astype(np.float32)

# Convert to class predictions
preds_cnn = classes[np.argmax(probs_cnn, axis = 1)]


# Calculate multiclass performance metrics

# Accuracy
accuracy_score(classes[np.argmax(y_test, axis = 1)], preds_cnn) 
# 0.6490384615384616

# Log loss
log_loss(y_test, probs_cnn) 
# 0.685978013122983


# Plot confusion matrix
plot_confusion(
  classes[np.argmax(y_test, axis = 1)], preds_cnn, classes, 
  "Recurrence plot + CNN classifier")

# Canadian weather data classification - PyTorch Lightning CNN implementation


import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# Define WeatherDataset classes

# TrainDataset takes in unprocessed inputs & targets
class TrainDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features & targets
  def __init__(self, x, y): 
    self.x = torch.tensor(x, dtype = torch.float32) # Store inputs
    self.y = torch.tensor(y, dtype = torch.float32) # Store targets
  
  # Return data length  
  def __len__(self):
    return len(self.x) 
  
  # Return a pair of features & target
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

# TestDataset takes in & returns unprocessed inputs only
class TestDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features & targets
  def __init__(self, x): 
    self.x = torch.tensor(x, dtype = torch.float32) # Store inputs
  
  # Return data length  
  def __len__(self):
    return len(self.x) 
  
  # Return a pair of features & target
  def __getitem__(self, idx):
    return self.x[idx]


# Define Lightning module
class CNN(L.LightningModule):
  
  # Initialize model
  def __init__(self, hyperparams_dict):
    
    # Delegate function to parent class
    super().__init__() 
    self.save_hyperparameters(logger = False) # Save external hyperparameters so
    # they are available when loading saved models
    
    # Define hyperparameters
    self.input_size = hyperparams_dict["input_size"]
    self.hidden_size = hyperparams_dict["hidden_size"]
    self.learning_rate = hyperparams_dict["learning_rate"]
    self.loss = torch.nn.CrossEntropyLoss()
    
    # Define architecture 
    
    
    # Initialize weights to conform with self-normalizing SELU activation
    for layer in self.encoder:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    

  # Define forward propagation
  def forward(self, x):
    output = self.network((x.view(x.size(0), -1)))
    return output
  
  
  # Define training loop
  def training_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return training loss
    x, y = batch
    output = self.forward(x)
    loss = self.loss(output, y)
    self.log(
      "train_loss", loss, 
      on_step = True, on_epoch = True, prog_bar = True, logger = True)
      
    return loss
  
  
  # Define validation loop
  def validation_step(self, batch, batch_idx):
    
    # Make predictions, calculate & log validation loss
    x, y = batch
    output = self.forward(x)
    loss = self.loss(output, y)
    self.log(
      "val_loss", loss, 
      on_step = True, on_epoch = True, prog_bar = True, logger = True)
      
    return loss
  
  
  # Define optimization algorithm, LR scheduler
  def configure_optimizers(self):
    
    # Adam optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    
    # Plateau LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer) 
    
    return {
    "optimizer": optimizer,
    "lr_scheduler": {
      "scheduler": lr_scheduler
      }
    }


# Create copy of Optuna callback with lightning.pytorch namespace as a workaround,
# as Optuna code uses pytorch_lightning namespace which causes an error
class OptunaPruning(PyTorchLightningPruningCallback, Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

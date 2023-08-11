# Canadian weather data classification - PyTorch Lightning CNN implementation


import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# Define WeatherDataset classes

# TrainDataset takes in preprocessed inputs & targets
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

# TestDataset takes in & returns preprocessed inputs only
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
    self.input_channels = hyperparams_dict["input_channels"]
    self.learning_rate = hyperparams_dict["learning_rate"]
    self.lr_decay = hyperparams_dict["lr_decay"]

    # Define architecture 
    self.network = torch.nn.Sequential(
  
      # Convolutional block 1: Conv layer + SELU activation + max pooling
      torch.nn.Conv2d(
        in_channels = self.input_channels, out_channels = 16, 
        kernel_size = 11, padding = 5), 
      torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
      # Output: (N, 16, 14, 14)
  
      # ConvBlock2
      torch.nn.Conv2d(
        in_channels = 16, out_channels = 32, kernel_size = 7, padding = 3), 
      torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
      # Output: (N, 32, 7, 7)
  
      # ConvBlock3
      torch.nn.Conv2d(
        in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2),
      torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
      # Output: (N, 64, 4, 4)
  
      # ConvBlock4
      torch.nn.Conv2d(
        in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1),
      torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
      # Output: (N, 3, 2, 2)
  
      # Global avg pooling + flatten
      torch.nn.AdaptiveAvgPool2d(output_size = 1),
      torch.nn.Flatten()
      # Output: (N, 3)
    ) 
    
    # Loss function
    self.loss = torch.nn.CrossEntropyLoss()
    
    # Softmax activation for probability predictions
    self.softmax = torch.nn.Softmax(dim = 1)
    
    # Initialize weights to conform with self-normalizing SELU activation
    for layer in self.network:
      if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    

  # Define forward propagation
  def forward(self, x):
    output = self.network(x)
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
  
  
  # Define prediction step to return probabilities, not logits
  def predict_step(self, batch, batch_idx):
    
    # Make predictions, apply softmax activation to get probabilities
    x = batch
    output = self.forward(x)
    pred = self.softmax(output)
    return pred
  
  
  # Define optimization algorithm, LR scheduler
  def configure_optimizers(self):
    
    # Adam optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    
    # Exponential LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
      optimizer, gamma = self.lr_decay) 
    
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


# # Experimenting with CNN design
# 
# # Generate a batch with input data dimensions
# x = torch.rand(size = (64, 8, 28, 28))
# x.shape
# 
# # NiN-like, 4 convolutional blocks, no dense layers
# # Convolutions reduce dimensions from 28x28 to 2x2 (/14), increase channels from
# # 8 to 64 (8x).
# network = torch.nn.Sequential(
#   
#   # Conv1
#   torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 11, padding = 5), 
#   torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
#   
#   # Conv2
#   torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 7, padding = 3),
#   torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
# 
#   # Conv3
#   torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2),
#   torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
# 
#   # Conv4
#   torch.nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1),
#   torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
# 
#   # # Global avg pooling + flatten, output of logits
#   # torch.nn.AdaptiveAvgPool2d(1),
#   # torch.nn.Flatten(),
#   # # torch.nn.Softmax(dim = 1)
# ) 
# 
# # Test run
# network(x)
# network(x).shape

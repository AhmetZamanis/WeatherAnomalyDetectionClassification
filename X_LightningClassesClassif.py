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

    # Define architecture (based on NiN architecture)
    self.network = torch.nn.Sequential(
      
      # Input dims: (N, 8, 28, 28)
  
      # Convolutional block 1: Main convolution of 11x11 + two 1x1 convs to add
      # local non-linearity + max pooling
      torch.nn.Conv2d(
            in_channels = self.input_channels, 
            out_channels = 16, kernel_size = 11, padding = 5), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 16, out_channels = 16, kernel_size = 1), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 16, out_channels = 16, kernel_size = 1), torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
      # Output dims: (N, 16, 13, 13)
          
      # Conv2: 7x7
      torch.nn.Conv2d(
        in_channels = 16, 
        out_channels = 32, kernel_size = 7, padding = 3), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 32, out_channels = 32, kernel_size = 1), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 32, out_channels = 32, kernel_size = 1), torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
      # Output dims: (N, 32, 6, 6)
      
      # Conv3: 5x5
      torch.nn.Conv2d(
        in_channels = 32, 
        out_channels = 64, kernel_size = 5, padding = 2), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 64, out_channels = 64, kernel_size = 1), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 64, out_channels = 64, kernel_size = 1), torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, stride = 1),
      # Output dims: (N, 64, 4, 4)
      
      # Conv4: 3x3 (reduces channel size to n. of target classes)
      torch.nn.Conv2d(
        in_channels = 64, 
        out_channels = 3, kernel_size = 3, padding = 2), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 3, out_channels = 3, kernel_size = 1), torch.nn.SELU(),
      torch.nn.Conv2d(
        in_channels = 3, out_channels = 3, kernel_size = 1), torch.nn.SELU(),
      torch.nn.MaxPool2d(kernel_size = 3, stride = 1),
      # Output dims: (N, 3, 4, 4)
      
      # Global avg. pooling + flatten
      torch.nn.AdaptiveAvgPool2d(output_size = 1),
      torch.nn.Flatten()
      # Output dims: (N, 3)
    ) 
    
    # Loss function
    self.loss = torch.nn.CrossEntropyLoss()
    
    # Softmax activation for probability predictions
    self.softmax = torch.nn.Softmax(dim = 1)
    
    # Initialize weights to conform with self-normalizing SELU activation
    # Cannot use Lazy layers in model architecture because the layer shapes have to
    # be determined before the weight initialization
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
# # Convolutions reduce dimensions from 28x28 to 4x4, increase channels from
# # 8 to 64 (8x).
# network = torch.nn.Sequential(
# 
#   # Conv1
#   torch.nn.Conv2d(
#         in_channels = 8, out_channels = 16,
#         kernel_size = 11, padding = 5), torch.nn.SELU(),
#    torch.nn.LazyConv2d(out_channels = 16, kernel_size = 1), torch.nn.SELU(),
#    torch.nn.LazyConv2d(out_channels = 16, kernel_size = 1), torch.nn.SELU(),
#    torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
#   # Output dims: (N, 16, 13, 13)
# 
#   # Conv2
#   torch.nn.LazyConv2d(out_channels = 32, kernel_size = 7, padding = 3), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 32, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 32, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
#   # Output dims: (N, 32, 6, 6)
# 
#   # Conv3
#   torch.nn.LazyConv2d(out_channels = 64, kernel_size = 5, padding = 2), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 64, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 64, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, stride = 1),
#   # Output dims: (N, 64, 4, 4)
# 
#   # Conv4 (reduces channel size to n. of target classes)
#   torch.nn.LazyConv2d(out_channels = 3, kernel_size = 3, padding = 2), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 3, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.LazyConv2d(out_channels = 3, kernel_size = 1), torch.nn.SELU(),
#   torch.nn.MaxPool2d(kernel_size = 3, stride = 1),
#   # Output dims: (N, 3, 4, 4)
# 
#   # Global avg. pooling + flatten
#   torch.nn.AdaptiveAvgPool2d(output_size = 1),
#   #torch.nn.Flatten()
#   # Output dims: (N, 3)
# )
# 
# 
# 
# # Test run
# network(x)
# network(x).shape

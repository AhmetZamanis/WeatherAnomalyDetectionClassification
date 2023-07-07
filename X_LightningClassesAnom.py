# Canadian weather data anomaly data - PyTorch Lightning autoencoder implementation


import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# Define WeatherDataset classes

# TrainDataset takes in preprocessed inputs, returns it both as input and target
class TrainDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features & targets
  def __init__(self, x): 
    self.x = torch.tensor(x, dtype = torch.float32) # Store inputs
    self.y = self.x # The inputs are also the targets
  
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
class AutoEncoder(L.LightningModule):
  
  # Initialize model
  def __init__(self, hyperparams_dict):
    
    # Delegate function to parent class
    super().__init__() 
    self.save_hyperparameters(logger = False) # Save external hyperparameters so
    # they are available when loading saved models
    
    # Define hyperparameters
    self.input_size = hyperparams_dict["input_size"]
    self.hidden_size = hyperparams_dict["hidden_size"]
    self.latent_size = hyperparams_dict["latent_size"]
    self.learning_rate = hyperparams_dict["learning_rate"]
    self.dropout = hyperparams_dict["dropout"]
    
    # Define architecture 
    
    # Encoder
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(self.input_size, self.hidden_size), # Hidden 1
      torch.nn.SELU(), # Activation 1
      torch.nn.AlphaDropout(self.dropout), # Dropout 1
      torch.nn.Linear(self.hidden_size, self.latent_size), # Hidden 2
      torch.nn.SELU(), # Activation 2
      torch.nn.AlphaDropout(self.dropout) # Dropout 2
    )
    
    # Decoder
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(self.latent_size, self.hidden_size), # Hidden 1
      torch.nn.SELU(), # Activation 1
      torch.nn.AlphaDropout(self.dropout), # Dropout 1
      torch.nn.Linear(self.hidden_size, self.input_size) # Hidden 2 (output, no activation)
    )
    
    # Initialize weights to conform with self-normalizing SELU activation
    for layer in self.encoder:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    
    for layer in self.decoder:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    
  # Define forward propagation to return latent space representations
  # Remember to manually put model into eval mode & disable grad when returning
  # z outside model training
  def forward(self, x):
    z = self.encoder((x.view(x.size(0), -1)))
    return z
  
  # Define reconstruction loss function
  def _reconstruction_loss(self, output, y):
    
    # A vector of losses with shape(batch_size, input_dim)
    loss = torch.nn.functional.mse_loss(output, y, reduction = "none")
    
    # Mean over the input dimensions, returning shape(1, batch_size)
    loss = torch.mean(loss, dim = 1)
    
    # Mean over batch elements, returning batch loss
    loss = torch.mean(loss)
    
    return loss
  
  # Define training loop
  def training_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return training loss
    x, y = batch
    z = self.forward(x) # Encode
    output = self.decoder(z)  # Decode
    loss = self._reconstruction_loss(output, y)
    self.log(
      "train_loss", loss, 
      on_step = True, on_epoch = True, prog_bar = True, logger = True)
      
    return loss
  
  # Define validation loop
  def validation_step(self, batch, batch_idx):
    
    # Make predictions, calculate & log validation loss
    x, y = batch
    z = self.forward(x) # Encode
    output = self.decoder(z)  # Decode
    loss = self._reconstruction_loss(output, y)
    self.log(
      "val_loss", loss, 
      on_step = True, on_epoch = True, prog_bar = True, logger = True)
    
    # # Retrieve & log current epoch
    # epoch = self.current_epoch
    # self.log("val_epoch", current_epoch)
      
    return loss
  
  # Define prediction loop (because forward only returns the embeddings)
  def predict_step(self, batch, batch_idx):
    
    # Make & return predictions
    x = batch
    z = self.forward(x) # Encode
    pred = self.decoder(z)  # Decode
    
    return pred
    
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

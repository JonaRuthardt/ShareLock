import torch
import torch.nn as nn
import pytorch_lightning as pl

class Projection(pl.LightningModule):
    def __init__(self, config, embedding_size, input_size):
        super().__init__()
        
        self.config = config.copy()
        self.normalize = self.config.normalize
        
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        
        if num_layers > 0:
            layers = []
            for i in range(num_layers - 1):
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                input_size = hidden_size
            layers.append(nn.Linear(input_size, embedding_size))
            self.projection = nn.Sequential(*layers)
        else:
            self.projection = nn.Identity()
            
    def forward(self, x):
        projection = self.projection(x)
        if self.normalize:
            projection = projection / projection.norm(dim=1, keepdim=True)
        return projection
"""
Projection head components for classification tasks.
"""
from typing import List, Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Flexible classification head for vision models.
    
    Supports multiple hidden layers, dropout, batch normalization,
    and different activation functions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rates: Optional[List[float]] = None,
        use_batchnorm: bool = True,
        activation: str = "gelu",
    ):
        """
        Initialize a custom classification head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (number of classes)
            dropout_rates: Optional list of dropout rates for each layer
            use_batchnorm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        # Set up activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu" or activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Configure dropout rates
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_dims)
        elif len(dropout_rates) != len(hidden_dims):
            raise ValueError("Number of dropout rates must match number of hidden layers")
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates, strict=False)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(act_fn)
            
            # Dropout
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Logits tensor of shape [batch_size, output_dim]
        """
        return self.mlp(x)
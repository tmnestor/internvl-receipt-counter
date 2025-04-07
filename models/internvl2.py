"""
InternVL2 model implementation for receipt counting.

This module adapts the InternVL2 vision-language model for the task
of receipt counting, using the vision encoder portion of the model.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import InternVL2ForVisionLanguageModeling, InternVL2VisionConfig

from models.components.projection_head import ClassificationHead


class InternVL2ReceiptClassifier(nn.Module):
    """
    InternVL2-based receipt classification model.
    
    Adapts the InternVL2 architecture for the receipt counting task by using only
    the vision encoder and adding a custom classification head.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        pretrained: bool = True,
        freeze_vision_encoder: bool = False,
    ):
        """
        Initialize the InternVL2 receipt classifier.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            freeze_vision_encoder: Whether to freeze the vision encoder
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the InternVL2 model
        pretrained_path = config["model"]["pretrained_path"]
        use_8bit = config["model"].get("use_8bit", False)
        
        # Verify the path exists
        if not Path(pretrained_path).exists():
            raise ValueError(f"Model path does not exist: {pretrained_path}. Please provide a valid path to the pre-downloaded model.")
        
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            self.model = InternVL2ForVisionLanguageModeling.from_pretrained(
                pretrained_path,
                load_in_8bit=use_8bit,
                device_map="auto",
                local_files_only=True  # Ensure no download attempts
            )
        else:
            # Load with default initialization (for debugging)
            vision_config = InternVL2VisionConfig.from_pretrained(
                pretrained_path, local_files_only=True
            )
            self.model = InternVL2ForVisionLanguageModeling.from_config(vision_config)
        
        # Extract vision encoder from the full model
        self.vision_encoder = self.model.vision_model
        
        # Remove language model-related components to save memory
        if hasattr(self.model, "language_model"):
            del self.model.language_model
        if hasattr(self.model, "text_hidden_fcs"):
            del self.model.text_hidden_fcs
            
        # Freeze vision encoder if required
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Get vision encoder output dimension
        vision_hidden_size = self.vision_encoder.config.hidden_size
        
        # Create a custom classification head
        self.classification_head = ClassificationHead(
            input_dim=vision_hidden_size,
            hidden_dims=config["model"]["classifier"]["hidden_dims"],
            output_dim=config["model"]["num_classes"],
            dropout_rates=config["model"]["classifier"]["dropout_rates"],
            use_batchnorm=config["model"]["classifier"]["batch_norm"],
            activation=config["model"]["classifier"]["activation"],
        )
        
    def unfreeze_vision_encoder(self, lr_multiplier: float = 0.1) -> List[Dict]:
        """
        Unfreeze the vision encoder and prepare parameter groups for optimizer.
        
        Args:
            lr_multiplier: Learning rate multiplier for vision encoder parameters
            
        Returns:
            List of parameter groups for optimizer
        """
        # Unfreeze all vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
            
        # Create parameter groups with different learning rates
        return [
            {'params': self.classification_head.parameters()},
            {'params': self.vision_encoder.parameters(), 'lr': lr_multiplier}
        ]
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            
        Returns:
            Dictionary with logits and other outputs
        """
        # Pass through vision encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        # Global average pooling over sequence dimension
        pooled_output = image_embeds.mean(dim=1)
        
        # Pass through classifier head
        logits = self.classification_head(pooled_output)
        
        return {
            "logits": logits,
            "embeddings": pooled_output
        }
    
    def get_attention_maps(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from the vision model for visualization.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            
        Returns:
            List of attention maps from each transformer block
        """
        # Enable output_attentions
        original_setting = self.vision_encoder.config.output_attentions
        self.vision_encoder.config.output_attentions = True
        
        # Forward pass
        outputs = self.vision_encoder(pixel_values=pixel_values, output_attentions=True)
        
        # Get attention weights
        attention_maps = outputs.attentions
        
        # Reset config to original setting
        self.vision_encoder.config.output_attentions = original_setting
        
        return attention_maps
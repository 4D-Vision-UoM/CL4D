"""
P4Transformer Encoder for LLaVA

Integrates 4D motion encoder (batch, frames, points, 3) with LLaVA.
This encoder processes point cloud sequences instead of images.
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import CLIPImageProcessor
from llava.model.multimodal_encoder.p4transformer_module.models import P4Transformer


@dataclass
class P4TransformerVisionConfig:
    """
    Configuration class for P4Transformer motion encoder.
    """

    model_name: str = "p4transformer-encoder"
    hidden_size: int = 1024  # Output feature dimension from p4transformer
    num_frames: int = 32  # Number of frames in sequence
    num_points: int = 2048  # Number of points per frame
    num_channels: int = 3  # xyz coordinates

    # P4Transformer specific configs (will be loaded from checkpoint)
    checkpoint_path: str = None
    config_path: str = None


class ConfigManager:
    """YAML configuration management."""

    @staticmethod
    def load_config(config_path: str):
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config


class P4TransformerVisionTower(nn.Module):
    """
    Vision tower wrapper for P4Transformer encoder.

    This replaces the standard image encoder with a 4D point cloud encoder.
    Input: (batch, frames, points, 3) instead of (batch, 3, height, width)
    """

    def __init__(self, vision_tower, args, delay_load=False):
        """
        Args:
            vision_tower: str, path to checkpoint directory (e.g., ""output/DynAction4DHuman/P4Transformer")
            args: model arguments containing configuration
            delay_load: bool, whether to delay model loading
        """
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower

        # Extract configuration from args
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.checkpoint_dir = Path(vision_tower)
        self.config_path = self.checkpoint_dir / "config.yaml"
        self.checkpoint_path = self.checkpoint_dir / "checkpoints" / "best.pth"

        config = ConfigManager.load_config(str(self.config_path))
        model_config = config["model"]
        print(f"Loaded P4Transformer config from {self.config_path}:")
        self.cfg_only = P4TransformerVisionConfig(
            checkpoint_path=str(self.checkpoint_path),
            config_path=str(self.config_path),
            hidden_size=model_config["pipeline_args"].get("dim"),
            num_frames=config["data"].get("num_frames"),
        )

        # Optionally load immediately
        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()

    def load_model(self, device_map=None):
        """
        Load the P4Transformer motion encoder from checkpoint.
        """
        if self.is_loaded:
            print(
                f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping."
            )
            return

        print(f"Loading P4Transformer motion encoder from: {self.checkpoint_dir}")

        config = ConfigManager.load_config(str(self.config_path))
        model_config = config["model"]

        # Get pipeline arguments from config
        pipeline_args = model_config["pipeline_args"]

        # Create pipeline instance with feature_mode='all' to get all tokens
        print("Creating P4Transformer pipeline...")
        self.vision_tower = P4Transformer(feature_mode="all", **pipeline_args)

        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Load model state dict
        if "model_state_dict" in checkpoint:
            missing_keys, unexpected_keys = self.vision_tower.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,  # Allow missing text encoder keys
            )

            if missing_keys:
                print(
                    f"⚠ Missing keys: {len(missing_keys)} (expected if text encoder removed)"
                )
            if unexpected_keys:
                print(f"⚠ Unexpected keys: {len(unexpected_keys)}")

            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Fallback: try loading directly
            self.vision_tower.load_state_dict(checkpoint, strict=False)
            print("✓ Loaded model (direct)")

        # Freeze the model (standard practice for vision towers)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        # Create a dummy image processor (not used, but required by LLaVA interface)
        # This is just for compatibility - we'll handle 4D data directly
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )

        # Verify total and trainable parameters
        total_params = sum(p.numel() for p in self.vision_tower.parameters())
        trainable_params = sum(
            p.numel() for p in self.vision_tower.parameters() if p.requires_grad
        )

        print(f"✓ P4Transformer encoder loaded:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen: {trainable_params == 0}")

        self.is_loaded = True

    def feature_select(self, motion_features):
        # keeping the code as same as clip. Feature selection not used
        """
        Select and process features from model output.

        For P4Transformer with feature_mode='all', the output is already in the format we need:
        (batch, num_tokens, hidden_size)
        """
        # P4Transformer with feature_mode='all' returns: [batch, num_tokens, hidden_size]
        # This is already in the correct format for LLaVA
        return motion_features

    @torch.no_grad()
    def forward(self, motion_data):
        """
        Forward pass through the P4Transformer motion encoder.

        Args:
            motion_data: torch.Tensor of shape (batch, frames, points, 3)
                       or list of such tensors

        Returns:
            motion_features: torch.Tensor of shape (batch, num_tokens, hidden_size)
        """
        motion_forward_outs = self.vision_tower(
            motion_data.to(device=self.device, dtype=self.dtype)
        )

        motion_features = self.feature_select(motion_forward_outs).to(motion_data.dtype)

        return motion_features

    # ========================================================================
    # Required Properties - LLaVA expects these to exist
    # ========================================================================

    @property
    def dummy_feature(self):
        """Return a dummy feature tensor for initialization."""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Return the dtype of the model."""
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        """Return the model configuration."""
        if self.is_loaded:
            # Return a config-like object with necessary attributes
            return self.cfg_only
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """Return the hidden size (output feature dimension)."""
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        """
        Return number of patches per side.
        For P4Transformer, this is derived from the ViT architecture.
        """
        # ViT processes the motion as [batch, 3, frames, feature_dim]
        # With patch_size=16, we get num_patches = (frames/patch_size) * (feature_dim/patch_size)
        # For simplicity, we approximate based on typical output
        return 14  # Typical for ViT base

    @property
    def num_patches(self):
        """
        Return total number of patches (tokens).

        For P4Transformer with feature_mode='all', this depends on the ViT configuration.
        Typically: num_patches = (num_frames / patch_size) * (feature_dim / patch_size)

        For a ViT-base with 32 frames and feature_dim=512, patch_size=16:
        num_patches = (32/16) * (512/16) = 2 * 32 = 64 tokens
        """
        # This will be overridden by actual output size at runtime
        # Just provide a reasonable default
        return 65  # Typical ViT-base (14*14 + 1 CLS token)


def build_p4transformer_vision_tower(vision_tower, args, **kwargs):
    """
    Factory function to build the P4Transformer vision tower.
    This is called by the builder.
    """
    return P4TransformerVisionTower(vision_tower, args=args, **kwargs)

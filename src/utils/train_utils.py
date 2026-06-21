import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
import numpy as np
from tqdm import tqdm


class TrainingLogger:
    """
    Comprehensive logging setup for training with both file and console output.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = None,
        log_level: str = "INFO",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save log files
            experiment_name: Name of experiment (auto-generated if None)
            log_level: Overall logging level
            console_level: Console output level
            file_level: File output level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"4d_text_alignment_{timestamp}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"

        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        simple_formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%H:%M:%S"
        )

        # File handler (detailed logging)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (simpler logging)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # Log initialization
        self.logger.info(f"Training logger initialized: {experiment_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Log levels - Console: {console_level}, File: {file_level}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def log_config(self, config: Dict[str, Any]):
        """Log configuration dictionary."""
        self.logger.info("Training Configuration:")
        self.logger.info("-" * 50)
        self._log_dict(config, indent=0)
        self.logger.info("-" * 50)

    def _log_dict(self, d: Dict[str, Any], indent: int = 0):
        """Recursively log dictionary contents."""
        for key, value in d.items():
            spacing = "  " * indent
            if isinstance(value, dict):
                self.logger.info(f"{spacing}{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.logger.info(f"{spacing}{key}: {value}")


class TensorBoardLogger:
    """
    TensorBoard integration for training metrics and visualization.
    """

    def __init__(self, log_dir: str, experiment_name: str, flush_secs: int = 60):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of experiment
            flush_secs: How often to flush logs to disk
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.tb_dir = self.log_dir / experiment_name

        self.writer = SummaryWriter(log_dir=str(self.tb_dir), flush_secs=flush_secs)

        # Track metrics for averaging
        self.metrics_buffer = defaultdict(list)

        print(f"TensorBoard logging to: {self.tb_dir}")
        print(f"View with: tensorboard --logdir {self.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        try:
            # Coerce torch tensors and numpy arrays to scalars if possible
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar_value = value.item()
                else:
                    # Reduce multi-element tensors to a single scalar (mean)
                    scalar_value = float(value.detach().cpu().numpy().mean())
            else:
                # Try numpy arrays
                try:
                    import numpy as _np

                    if isinstance(value, _np.ndarray):
                        if value.size == 1:
                            scalar_value = float(value.item())
                        else:
                            scalar_value = float(value.mean())
                    else:
                        scalar_value = float(value)
                except Exception:
                    # Fallback: try cast to float
                    scalar_value = float(value)

            self.writer.add_scalar(tag, scalar_value, step)
        except Exception as e:
            print(f"Warning: Failed to log scalar '{tag}' at step {step}: {e}")

    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: int):
        """Log multiple related scalars."""
        self.writer.add_scalars(tag, value_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of tensor values."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """Log image tensor."""
        self.writer.add_image(tag, img_tensor, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)

    def log_hyperparameters(
        self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]
    ):
        """Log hyperparameters and metrics."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        lr: float = None,
        prefix: str = "train",
    ):
        """Log training metrics for an epoch."""
        try:
            self.log_scalar(f"{prefix}/loss", loss, epoch)
        except Exception as e:
            print(f"Warning: Could not log {prefix}/loss: {e}")

        if lr is not None:
            try:
                self.log_scalar(f"{prefix}/learning_rate", lr, epoch)
            except Exception as e:
                print(f"Warning: Could not log {prefix}/learning_rate: {e}")

        for metric_name, metric_value in metrics.items():
            try:
                self.log_scalar(f"{prefix}/{metric_name}", metric_value, epoch)
            except Exception as e:
                try:
                    if (
                        isinstance(metric_value, torch.Tensor)
                        and metric_value.numel() > 1
                    ):
                        reduced = float(metric_value.detach().cpu().numpy().mean())
                        self.log_scalar(f"{prefix}/{metric_name}", reduced, epoch)
                    else:
                        print(f"Warning: Could not log metric {metric_name}: {e}")
                except Exception as e2:
                    print(f"Warning: Failed to coerce metric {metric_name}: {e2}")

    def log_validation_metrics(
        self, epoch: int, loss: float, metrics: Dict[str, float]
    ):
        """Log validation metrics for an epoch."""
        try:
            self.log_training_metrics(epoch, loss, metrics, prefix="val")
        except Exception as e:
            print(f"Warning: Failed to log validation metrics at epoch {epoch}: {e}")

    def accumulate_metric(self, tag: str, value: float):
        """Accumulate metric for later averaging."""
        self.metrics_buffer[tag].append(value)

    def log_accumulated_metrics(self, step: int, prefix: str = ""):
        """Log accumulated metrics as averages and clear buffer."""
        for tag, values in self.metrics_buffer.items():
            if values:
                avg_value = np.mean(values)
                full_tag = f"{prefix}/{tag}" if prefix else tag
                self.log_scalar(full_tag, avg_value, step)

        # Clear buffer
        self.metrics_buffer.clear()

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class CheckpointManager:
    """
    Comprehensive checkpoint management for training.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: int = None,
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of experiment
            save_best: Whether to save best model
            save_last: Whether to save last model
            save_every_n_epochs: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.max_checkpoints = max_checkpoints

        # Track best metrics
        self.best_metric_value = None
        self.best_epoch = None
        self.metric_mode = "min"  # 'min' for loss, 'max' for accuracy

        # Track saved checkpoints
        self.saved_checkpoints = []

        print(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def set_metric_mode(self, mode: str):
        """Set metric mode ('min' for loss, 'max' for accuracy)."""
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'")
        self.metric_mode = mode

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric_value: float = None,
        is_best: bool = False,
        filename: str = None,
    ) -> str:
        """
        Save checkpoint.

        Args:
            state_dict: State dictionary to save
            epoch: Current epoch
            metric_value: Metric value for best model tracking
            is_best: Whether this is the best model
            filename: Custom filename (optional)

        Returns:
            Path to saved checkpoint
        """

        # Determine if this is the best model
        if metric_value is not None and self.save_best:
            if self.best_metric_value is None:
                is_best = True
            elif self.metric_mode == "min" and metric_value < self.best_metric_value:
                is_best = True
            elif self.metric_mode == "max" and metric_value > self.best_metric_value:
                is_best = True

            if is_best:
                self.best_metric_value = metric_value
                self.best_epoch = epoch

        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "state_dict": state_dict,
            "metric_value": metric_value,
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
        }

        # Determine filename
        if filename is None:
            if is_best and self.save_best:
                filename = "best.pth"
            else:
                filename = f"epoch_{epoch:03d}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Update saved checkpoints list
        if not is_best:  # Don't count best model in regular checkpoint limit
            self.saved_checkpoints.append(checkpoint_path)

            # Remove old checkpoints if exceeding limit
            if len(self.saved_checkpoints) > self.max_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

        # Save last model if requested
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pth"
            torch.save(checkpoint_data, last_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded checkpoint data
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Update internal state
        self.best_metric_value = checkpoint.get("best_metric_value")
        self.best_epoch = checkpoint.get("best_epoch")

        return checkpoint

    def get_best_checkpoint_path(self) -> str:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best.pth"
        if best_path.exists():
            return str(best_path)
        return None

    def get_last_checkpoint_path(self) -> str:
        """Get path to last checkpoint."""
        last_path = self.checkpoint_dir / "last.pth"
        if last_path.exists():
            return str(last_path)
        return None

    def should_save_checkpoint(self, epoch: int) -> bool:
        """Determine if checkpoint should be saved this epoch."""
        if self.save_every_n_epochs is None:
            return False
        return epoch % self.save_every_n_epochs == 0


class TrainingResumer:
    """
    Handle training resume from checkpoints.
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize training resumer.

        Args:
            checkpoint_manager: CheckpointManager instance
        """
        self.checkpoint_manager = checkpoint_manager

    def resume_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        checkpoint_path: str = None,
        resume_best: bool = False,
        resume_last: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Resume training from checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint path (optional)
            resume_best: Resume from best checkpoint
            resume_last: Resume from last checkpoint (default)

        Returns:
            Tuple of (start_epoch, training_info)
        """

        # Determine checkpoint path
        if checkpoint_path is None:
            if resume_best:
                checkpoint_path = self.checkpoint_manager.get_best_checkpoint_path()
            elif resume_last:
                checkpoint_path = self.checkpoint_manager.get_last_checkpoint_path()

            if checkpoint_path is None:
                print("No checkpoint found for resuming. Starting from epoch 0.")
                return 0, {}

        print(f"Resuming training from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Load model state
        if "model_state_dict" in checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"]["model_state_dict"])
        elif "model" in checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"]["model"])
        else:
            print("Warning: No model state found in checkpoint")

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint["state_dict"]:
            optimizer.load_state_dict(checkpoint["state_dict"]["optimizer_state_dict"])
        elif "optimizer" in checkpoint["state_dict"]:
            optimizer.load_state_dict(checkpoint["state_dict"]["optimizer"])
        else:
            print("Warning: No optimizer state found in checkpoint")

        # Load scheduler state
        if scheduler is not None:
            if "scheduler_state_dict" in checkpoint["state_dict"]:
                scheduler.load_state_dict(
                    checkpoint["state_dict"]["scheduler_state_dict"]
                )
            elif "scheduler" in checkpoint["state_dict"]:
                scheduler.load_state_dict(checkpoint["state_dict"]["scheduler"])
            else:
                print("Warning: No scheduler state found in checkpoint")

        start_epoch = checkpoint["epoch"] + 1

        training_info = {
            "resumed_from_epoch": checkpoint["epoch"],
            "best_metric_value": checkpoint.get("best_metric_value"),
            "best_epoch": checkpoint.get("best_epoch"),
            "experiment_name": checkpoint.get("experiment_name"),
            "resume_timestamp": datetime.now().isoformat(),
        }

        print(f"Resumed training from epoch {checkpoint['epoch']}")
        if training_info["best_metric_value"] is not None:
            print(
                f"Best metric so far: {training_info['best_metric_value']} at epoch {training_info['best_epoch']}"
            )

        return start_epoch, training_info


class ConfigManager:
    """
    YAML configuration management for training.
    """

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


class MetricsTracker:
    """
    Track and compute training metrics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of sliding window for moving averages
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)

    def get_moving_average(self, metric_name: str) -> float:
        """Get moving average of metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return np.mean(list(self.metrics[metric_name]))

    def get_current_value(self, metric_name: str) -> float:
        """Get most recent value of metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return self.metrics[metric_name][-1]

    def reset_epoch(self):
        """Reset metrics for new epoch."""
        for key, values in self.metrics.items():
            if values:
                self.epoch_metrics[key].append(np.mean(list(values)))
        self.metrics.clear()

    def get_epoch_summary(self) -> Dict[str, float]:
        """Get summary of current epoch metrics."""
        return {
            key: np.mean(list(values)) for key, values in self.metrics.items() if values
        }


def setup_training_environment(
    config: Dict[str, Any], experiment_name: str = None
) -> Tuple[TrainingLogger, TensorBoardLogger, CheckpointManager]:
    """
    Setup complete training environment with logging, tensorboard, and checkpointing.

    Args:
        config: Training configuration
        experiment_name: Experiment name (auto-generated if None)

    Returns:
        Tuple of (logger, tb_logger, checkpoint_manager)
    """

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = config.get("experiment", {}).get("name")
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"4d_text_alignment_{timestamp}"

    # Setup paths
    output_dir = (
        Path(config.get("paths", {}).get("output_dir", "output")) / experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / config.get("paths", {}).get("log_dir", "logs")
    checkpoint_dir = output_dir / config.get("paths", {}).get(
        "checkpoint_dir", "checkpoints"
    )
    tensorboard_dir = output_dir / config.get("paths", {}).get(
        "tensorboard_dir", "tensorboard"
    )

    # Save config to output directory
    config_save_path = output_dir / "config.yaml"
    ConfigManager.save_config(config, str(config_save_path))

    # Setup logging
    log_config = config.get("logging", {})
    logger = TrainingLogger(
        log_dir=str(log_dir),
        experiment_name=experiment_name,
        log_level=log_config.get("log_level", "INFO"),
        console_level=log_config.get("console_level", "INFO"),
        file_level=log_config.get("file_level", "DEBUG"),
    )

    # Setup TensorBoard
    tb_logger = TensorBoardLogger(
        log_dir=str(tensorboard_dir),
        experiment_name=experiment_name,
        flush_secs=log_config.get("tensorboard_flush_secs", 60),
    )

    # Setup checkpointing
    ckpt_config = config.get("checkpointing", {})
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        experiment_name=experiment_name,
        save_best=ckpt_config.get("save_best", True),
        save_last=ckpt_config.get("save_last", True),
        save_every_n_epochs=ckpt_config.get("save_every_n_epochs"),
        max_checkpoints=ckpt_config.get("max_checkpoints", 5),
    )

    # Set metric mode for best model tracking
    val_config = config.get("training", {}).get("validation", {})
    metric_mode = val_config.get("mode", "min")
    checkpoint_manager.set_metric_mode(metric_mode)

    # Log configuration
    logger.log_config(config)
    logger.info(f"Training environment setup complete: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    return logger, tb_logger, checkpoint_manager


# Convenience function for easy import
def create_training_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    additional_state: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Create training state dictionary for checkpointing.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save (optional)
        additional_state: Additional state to save

    Returns:
        State dictionary
    """
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    if additional_state is not None:
        state_dict.update(additional_state)

    return state_dict


class ValidationEvaluator:
    """
    Comprehensive validation evaluator for text-to-scene and scene-to-text retrieval.

    Computes both batch-wise and global retrieval metrics with proper memory management.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        k_values: list = [1, 2, 3, 5, 10],
        text_encode_fn=None,
    ):
        """
        Initialize validation evaluator.

        Args:
            model: The model to evaluate
            criterion: Loss criterion
            device: Device to use for evaluation
            k_values: List of k values for Recall@k computation
            text_encode_fn: Function to encode text (callable that takes list of strings)
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.k_values = k_values
        self.text_encode_fn = text_encode_fn

    def compute_recall_at_k(
        self, sim_matrix: torch.Tensor, k: int, mode: str = "text_to_scene"
    ) -> float:
        """
        Compute Recall@k from similarity matrix.

        Args:
            sim_matrix: Similarity matrix [num_texts, num_scenes]
            k: Top k value
            mode: 'text_to_scene' or 'scene_to_text'

        Returns:
            Recall@k value
        """
        if mode == "text_to_scene":
            # For each text, check if correct scene is in top-k
            correct = 0
            num = sim_matrix.shape[0]
            for i in range(num):
                ranks = torch.argsort(sim_matrix[i], descending=True)
                if i in ranks[:k]:
                    correct += 1
            return correct / num if num > 0 else 0.0

        elif mode == "scene_to_text":
            # For each scene, check if correct text is in top-k
            correct = 0
            num = sim_matrix.shape[1]
            for j in range(num):
                ranks = torch.argsort(sim_matrix[:, j], descending=True)
                if j in ranks[:k]:
                    correct += 1
            return correct / num if num > 0 else 0.0

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'text_to_scene' or 'scene_to_text'"
            )

    def evaluate_batch(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a single batch.

        Args:
            batch: Batch data

        Returns:
            Tuple of (loss, recall_metrics)
        """
        # Forward pass - get temporal features
        temporal_features = self.model(batch)

        # Encode text
        _, captions = batch
        if self.text_encode_fn is not None:
            text_features = self.text_encode_fn(captions)
        else:
            raise ValueError("text_encode_fn must be provided to ValidationEvaluator")

        # Compute loss
        loss = self.criterion(temporal_features, text_features)

        # Normalize embeddings for cosine similarity
        scene_emb = torch.nn.functional.normalize(temporal_features.cpu(), dim=1)
        text_emb = torch.nn.functional.normalize(text_features.cpu(), dim=1)

        # Compute similarity matrix: [num_texts, num_scenes]
        sim = text_emb @ scene_emb.t()

        # Compute recall metrics
        recall_metrics = {}
        for k in self.k_values:
            recall_metrics[f"recall@{k}_text_to_scene"] = self.compute_recall_at_k(
                sim, k, "text_to_scene"
            )
            recall_metrics[f"recall@{k}_scene_to_text"] = self.compute_recall_at_k(
                sim, k, "scene_to_text"
            )

        # Clean up to save memory
        del scene_emb, text_emb, sim, temporal_features, text_features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return loss.item(), recall_metrics

    def evaluate_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        logger: TrainingLogger = None,
        compute_global: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate entire dataloader with both batch-wise and global metrics.

        Args:
            dataloader: DataLoader to evaluate
            logger: Logger for error messages (optional)
            compute_global: Whether to compute global retrieval metrics

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Initialize metrics
        val_metrics = {"loss": 0.0, "num_sequences": 0, "num_batches": 0}

        # Batch-wise recall sums
        recall_sums = {
            "text_to_scene": {k: 0.0 for k in self.k_values},
            "scene_to_text": {k: 0.0 for k in self.k_values},
        }

        # Global embeddings (if computing global metrics)
        if compute_global:
            all_scene_embeddings = []
            all_text_embeddings = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
                try:
                    clips, captions = batch
                    clips = clips.to(self.device)
                    batch_size = clips.shape[0]

                    # Forward pass - get temporal features
                    temporal_features = self.model(clips)

                    # Encode text
                    if self.text_encode_fn is not None:
                        text_features = self.text_encode_fn(captions)
                    else:
                        raise ValueError(
                            "text_encode_fn must be provided to ValidationEvaluator"
                        )

                    # Compute loss
                    loss = self.criterion(temporal_features, text_features)
                    val_metrics["loss"] += loss.item() * batch_size
                    val_metrics["num_sequences"] += batch_size

                    # Normalize embeddings
                    scene_emb = torch.nn.functional.normalize(
                        temporal_features.cpu(), dim=1
                    )
                    text_emb = torch.nn.functional.normalize(text_features.cpu(), dim=1)

                    # Store for global computation
                    if compute_global:
                        all_scene_embeddings.append(scene_emb)
                        all_text_embeddings.append(text_emb)

                    # Compute batch-wise similarity matrix
                    sim = text_emb @ scene_emb.t()

                    # Compute batch-wise recall metrics
                    for k in self.k_values:
                        r_t2s = self.compute_recall_at_k(sim, k, "text_to_scene")
                        r_s2t = self.compute_recall_at_k(sim, k, "scene_to_text")
                        recall_sums["text_to_scene"][k] += r_t2s
                        recall_sums["scene_to_text"][k] += r_s2t

                    val_metrics["num_batches"] += 1

                    # Clean up batch-level tensors
                    del scene_emb, text_emb, sim, temporal_features, text_features, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    if logger:
                        logger.error(f"Error in validation batch {batch_idx}: {e}")
                    else:
                        print(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Compute averaged metrics
        if val_metrics["num_sequences"] > 0:
            val_metrics["loss"] /= val_metrics["num_sequences"]

        # Batch-wise average recalls
        if val_metrics["num_batches"] > 0:
            for k in self.k_values:
                val_metrics[f"recall@{k}_text_to_scene"] = (
                    recall_sums["text_to_scene"][k] / val_metrics["num_batches"]
                )
                val_metrics[f"recall@{k}_scene_to_text"] = (
                    recall_sums["scene_to_text"][k] / val_metrics["num_batches"]
                )
        else:
            for k in self.k_values:
                val_metrics[f"recall@{k}_text_to_scene"] = 0.0
                val_metrics[f"recall@{k}_scene_to_text"] = 0.0

        # Compute global retrieval metrics
        if compute_global and len(all_scene_embeddings) > 0:
            try:
                # Concatenate all embeddings
                global_scene_emb = torch.cat(all_scene_embeddings, dim=0)
                global_text_emb = torch.cat(all_text_embeddings, dim=0)

                # Compute global similarity matrix
                global_sim = global_text_emb @ global_scene_emb.t()

                # Compute global recall metrics
                for k in self.k_values:
                    val_metrics[f"global_recall@{k}_text_to_scene"] = (
                        self.compute_recall_at_k(global_sim, k, "text_to_scene")
                    )
                    val_metrics[f"global_recall@{k}_scene_to_text"] = (
                        self.compute_recall_at_k(global_sim, k, "scene_to_text")
                    )

                # Clean up global tensors
                del global_scene_emb, global_text_emb, global_sim
                del all_scene_embeddings, all_text_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                if logger:
                    logger.error(f"Error computing global metrics: {e}")
                else:
                    print(f"Error computing global metrics: {e}")
                # Set global metrics to 0 if computation fails
                for k in self.k_values:
                    val_metrics[f"global_recall@{k}_text_to_scene"] = 0.0
                    val_metrics[f"global_recall@{k}_scene_to_text"] = 0.0

        return val_metrics

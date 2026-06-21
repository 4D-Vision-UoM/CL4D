import os
import sys
import torch
from llava.train.train import train


################################################################################
# CONFIGURATION
################################################################################
class Config:
    """Training configuration"""

    # Base paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model configuration
    MODEL_NAME = "liuhaotian/llava-v1.5-7b"
    PRETRAINED_4D_CHECKPOINT = "output/DynAction4DHuman/CL4D"
    MM_VISION_TOWER_TYPE = (
        "cl4d"  # Options: cl4d, motionpointnet, psttransformer, p4transformer
    )

    # Data paths
    DATA_PATH = os.path.join(SCRIPT_DIR, "data")
    VQA_PATH = "data/DynAction4D_VQA"
    MOTION_PATH = "data/DynAction4D_Human"

    # Output configuration
    OUTPUT_DIR = os.path.join(
        SCRIPT_DIR, "output/Llava/CL4D_DynAction4D_VQA"
    )

    # Training hyperparameters
    NUM_EPOCHS = 1
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-3
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.0
    MAX_LENGTH = 2048
    NUM_WORKERS = 24

    # Training strategy
    FREEZE_BACKBONE = True
    TUNE_MM_MLP_ADAPTER = True
    SAVE_TOTAL_LIMIT = 3
    LOGGING_STEPS = 100


def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("LLaVA Training with 4D Motion Encoder")
    print("=" * 80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Vision Tower Type: {config.MM_VISION_TOWER_TYPE}")
    print(f"Pretrained 4D checkpoint: {config.PRETRAINED_4D_CHECKPOINT}")
    print(f"Base data path: {config.DATA_PATH}")
    print(f"VQA path: {config.VQA_PATH}")
    print(f"Motion path: {config.MOTION_PATH}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Prepare training arguments
    training_args = [
        # Model configuration
        "--model_name_or_path",
        config.MODEL_NAME,
        "--version",
        "v1",
        "--vision_tower",
        config.PRETRAINED_4D_CHECKPOINT,
        "--mm_vision_tower_type",
        config.MM_VISION_TOWER_TYPE,
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-1",
        # Data configuration
        "--data_path",
        config.DATA_PATH,
        "--vqa_path",
        config.VQA_PATH,
        "--motion_path",
        config.MOTION_PATH,
        "--use_motion_data",
        "True",
        # Vision processing
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "pad",
        "--group_by_modality_length",
        "True",
        # Model freezing strategy
        "--freeze_backbone",
        str(config.FREEZE_BACKBONE),
        "--tune_mm_mlp_adapter",
        str(config.TUNE_MM_MLP_ADAPTER),
        # Training hyperparameters
        "--num_train_epochs",
        str(config.NUM_EPOCHS),
        "--per_device_train_batch_size",
        str(config.TRAIN_BATCH_SIZE),
        "--per_device_eval_batch_size",
        str(config.EVAL_BATCH_SIZE),
        "--gradient_accumulation_steps",
        str(config.GRADIENT_ACCUMULATION_STEPS),
        "--learning_rate",
        str(config.LEARNING_RATE),
        "--weight_decay",
        str(config.WEIGHT_DECAY),
        "--warmup_ratio",
        str(config.WARMUP_RATIO),
        "--lr_scheduler_type",
        "cosine",
        "--model_max_length",
        str(config.MAX_LENGTH),
        # Evaluation and saving
        "--evaluation_strategy",
        "epoch",
        "--save_strategy",
        "epoch",
        "--save_total_limit",
        str(config.SAVE_TOTAL_LIMIT),
        "--load_best_model_at_end",
        "True",
        "--metric_for_best_model",
        "eval_loss",
        "--greater_is_better",
        "False",
        # Output and logging
        "--output_dir",
        config.OUTPUT_DIR,
        "--logging_steps",
        str(config.LOGGING_STEPS),
        "--report_to",
        "tensorboard",
        # Performance optimizations
        "--bf16",
        "True" if torch.cuda.is_bf16_supported() else "False",
        "--fp16",
        "False" if torch.cuda.is_bf16_supported() else "True",
        "--tf32",
        "True" if torch.cuda.is_available() else "False",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        str(config.NUM_WORKERS),
        "--lazy_preprocess",
        "True",
    ]

    print("\nStarting training with CL4D motion encoder...")
    print("=" * 80)

    sys.argv = ["train"] + training_args
    train()

    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print(f"✓ Model saved to: {config.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

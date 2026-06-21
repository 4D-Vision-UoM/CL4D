# CL4D: Contrastive Language–4D Pretraining for Vision-Language Reasoning in Dynamic Scenes

> **A complete pipeline for pre-training CL4D motion encoders and 4D vision-language models** 🎯

This repository provides a comprehensive framework for training and evaluating 4D motion encoders including **CL4D**, MotionPointNet, P4Transformer, and PSTTransformer.

---

## 📑 Table of Contents
1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Pretrain 4D Motion Encoders](#pretrain-4d-motion-encoders)
4. [Train Vision-Language Model](#train-vision-language-model)
5. [Evaluation](#evaluation)

---

## 🛠️ Setup

### 1️⃣ Environment Setup

**For 4D Motion Encoder Training (CL4D):**
```bash
# Create conda environment
conda create -n cl4d python=3.10
conda activate cl4d

# Install dependencies
pip install -r requirements.txt
```

**For Vision-Language Model Training:** 
```bash
# Create LLaVA environment
conda create -n llava python=3.10
conda activate llava

# Install LLaVA dependencies
pip install --upgrade pip
pip install -e .

# Install flash-attention (optional, for faster training) 
pip install flash-attn --no-build-isolation
```

### 2️⃣ Additional Encoder Setups

For training with other encoder architectures, follow their respective setup instructions:

- 🔹 **MotionPointNet**: https://github.com/zhh6425/MotionPointNet
- 🔹 **P4Transformer**: https://github.com/hehefan/P4Transformer
- 🔹 **PST-Transformer**: https://github.com/hehefan/PST-Transformer

### 3️⃣ Download Pretrained Weights

**PointNet Weights:** 
```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download PointNet weights
gdown 1Fh061dVtq6_vzjmgFH2qX72C9y47N1cA -O weights/dVAE.pth
```

📎 Direct link: https://drive.google.com/file/d/1Fh061dVtq6_vzjmgFH2qX72C9y47N1cA/view

---

## 📊 Data Preparation

The CL4D pretraining requires the following datasets organized in the `data/` directory:

```
data/
├── DynAction4D_Human/       # Human motion sequences
│   ├── train/
│   └── test/
├── DynAction4D_Clutter/     # Cluttered scene human motion
│   ├── train/
│   └── test/
├── DynAction4D_Obj/         # Object-interactions
│   ├── train/
│   └── test/
└── DynAction4D_VQA/         # VQA dataset
    ├── train/
    └── test/
```

---

## 🎓 Pretrain 4D Motion Encoders

Pretraining phase trains the 4D motion encoder to align motion sequences with text descriptions using contrastive learning. 

###  Quick Start

```bash
# Activate environment
conda activate cl4d

# Train CL4D encoder on DynAction4D_Human dataset (default)
python train.py

# Train with specific configuration
python train.py --config configs/DynAction4D_Human/cl4d_config.yaml
```

###  Training Commands

**DynAction4D_Human:** 
```bash
# CL4D pipeline
python train.py --config configs/DynAction4D_Human/cl4d_config.yaml

# MotionPointNet pipeline
python train.py --config configs/DynAction4D_Human/motionpointnet_config.yaml

# P4Transformer pipeline
python train.py --config configs/DynAction4D_Human/p4transformer_config.yaml

# PSTTransformer pipeline
python train.py --config configs/DynAction4D_Human/psttransformer_config.yaml
```

**DynAction4D_Obj:** 
```bash
python train.py --config configs/DynAction4D_Obj/cl4d_config.yaml
python train.py --config configs/DynAction4D_Obj/motionpointnet_config.yaml
python train.py --config configs/DynAction4D_Obj/p4transformer_config.yaml
python train.py --config configs/DynAction4D_Obj/psttransformer_config.yaml
```

**DynAction4D_Clutter:** 
```bash
python train.py --config configs/DynAction4D_Clutter/cl4d_config.yaml
python train.py --config configs/DynAction4D_Clutter/motionpointnet_config.yaml
python train.py --config configs/DynAction4D_Clutter/p4transformer_config.yaml
python train.py --config configs/DynAction4D_Clutter/psttransformer_config.yaml
```

###  Training Arguments

-  `--config`: Path to configuration YAML file
-  `--resume`: Resume training from last checkpoint
-  `--experiment-name`: Custom experiment name (overrides config)

###  Configuration Files

Each config file specifies:
-  **Model architecture**: Encoder type and parameters
-  **Training parameters**: Batch size, learning rate, epochs
-  **Data parameters**: Dataset path, number of frames, points per frame
-  **Loss function**: CLIP or SigLIP contrastive loss

**Example config structure:**
```yaml
experiment:
  name: "DynAction4D_Human_CL4D_baseline"

data:
  data_root: "data/DynAction4D_Human"
  num_frames: 32
  num_points: 2048
  batch_size: 72

model:
  pipeline_class: "CL4DPipeline"
  projection_dims: 256
  
training:
  num_epochs: 100
  learning_rate: 0.0001
  loss_type: "clip"
```

### 📊 Monitoring Training

**TensorBoard:** 
```bash
tensorboard --logdir output/[experiment_name]/tensorboard
```

**Console Output:** 
-  Epoch progress with loss values
-  Training/validation metrics (text-motion retrieval)
-  Learning rate schedules
-  Best checkpoint

###  Output Structure

```
output/
└── [experiment_name]/
    ├── checkpoints/
    │   ├── best_model.pth         # Best model based on validation
    │   ├── last_model.pth         # Latest checkpoint
    │   └── epoch_*.pth            # Periodic checkpoints
    ├── logs/
    │   └── training.log           # Detailed training logs
    └── tensorboard/
        └── events.out.tfevents.*  # TensorBoard logs
```

---

## 🧠 Train Vision-Language Model

Fine-tunes LLaVA with the pretrained 4D motion encoder.

###  Quick Start

```bash
# Activate LLaVA environment
conda activate llava

# Train VLM with default settings
python train_vlm.py
```

###  Configuration

Edit the `Config` class in [train_vlm.py](train_vlm.py) to customize training:

```python
class Config:
    # Model paths
    MODEL_NAME = "liuhaotian/llava-v1.5-7b"
    PRETRAINED_4D_CHECKPOINT = "path to pretrained CL4D output folder"
    MM_VISION_TOWER_TYPE = "cl4d"  # cl4d, motionpointnet, psttransformer, p4transformer
    
    # Data paths
    VQA_PATH = "data/DynAction4D_VQA"
    MOTION_PATH = "data/DynAction4D_Human"
    
    # Training hyperparameters
    NUM_EPOCHS = 10
    TRAIN_BATCH_SIZE = 2
    LEARNING_RATE = 1e-3
    
    # Training strategy
    FREEZE_BACKBONE = True        # Freeze LLM
    TUNE_MM_MLP_ADAPTER = True    # Train only projection layer
```

###  Output Structure

```
output/llava/
└── [experiment_name]/
    ├── checkpoint-[step]/
    │   ├── config.json
    │   ├── mm_projector.bin      # Trained projection layer
    │   ├── pytorch_model.bin     
    │   └── trainer_state.json
    └── tensorboard/
        └── events.out.tfevents.*
```


---

## 📈 Evaluation

###  Evaluate VLM on VQA Task

```bash
# Run evaluation
python eval_vlm.py
```

### Configuration

Edit the `Config` class in [eval_vlm.py](eval_vlm.py):

```python
class Config:
    # Model paths
    MODEL_PATH = "Path to finetuned llava model"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"
    
    # Data paths
    VQA_PATH = "data/DynAction4D_VQA"
    MOTION_PATH = "data/DynAction4D_Human"
    
    # Generation parameters
    MAX_NEW_TOKENS = 128
    DO_SAMPLE = False
```

###  Output

Results are saved to:
```
output/[model_name]_evaluation_[timestamp].json
```

Format:
```json
[
  {
    "sample_idx": 0,
    "qa_idx": 0,
    "motion_id": "sequence_000123",
    "question": "What action is being performed?",
    "ground_truth": "Walking forward",
    "prediction": "The person is walking forward",
    "total_frames": 32
  }
]
```

---

## 📄 License

MIT License - see LICENSE file for details

---



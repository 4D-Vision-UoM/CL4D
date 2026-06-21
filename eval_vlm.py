# Evaluate fine-tuned model on motion VQA dataset

import os
import torch
import sys
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import ModelArguments
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from src.datasets.DynAction4D_VQA import MotionLazySupervisedDataset


################################################################################
# CONFIGURATION
################################################################################
class Config:
    """Evaluation configuration"""

    # Model paths
    MODEL_PATH = "output/Llava/CL4D_DynAction4D_VQA"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"

    # Data paths
    DATA_PATH = "data"
    VQA_PATH = "data/DynAction4D_VQA"
    MOTION_PATH = "data/DynAction4D_Human"

    # Output configuration
    OUTPUT_DIR = "output/Llava/CL4D_DynAction4D_VQA_evaluation"

    # Dataset parameters
    NUM_FRAMES = 32
    NUM_POINTS = 2048
    DATA_SPLIT = "test"

    # Generation parameters
    MAX_NEW_TOKENS = 128
    DO_SAMPLE = False
    CONVERSATION_VERSION = "v1"

    # Performance settings
    TORCH_DTYPE = torch.float16
    USE_CACHE = True
    CACHE_CLEAR_INTERVAL = 10  # Clear CUDA cache every N samples


class DataArgs:
    """Data loading arguments"""

    def __init__(self, config):
        self.data_path = config.DATA_PATH
        self.vqa_path = config.VQA_PATH
        self.motion_path = config.MOTION_PATH
        self.is_multimodal = True
        self.image_aspect_ratio = "pad"
        self.image_grid_pinpoints = None


################################################################################
# MODEL LOADING
################################################################################
def load_model(config):
    """Load fine-tuned model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BASE, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading config...")
    model_config = AutoConfig.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    vision_tower_path = model_config.mm_vision_tower

    print("Creating model with fine-tuned projector...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        config.MODEL_BASE,
        torch_dtype=config.TORCH_DTYPE,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = config.USE_CACHE

    # Initialize vision modules
    model_args = ModelArguments()
    model_args.vision_tower = vision_tower_path
    model_args.mm_vision_tower_type = getattr(
        model_config, "mm_vision_tower_type", None
    )
    model_args.mm_vision_select_layer = model_config.mm_vision_select_layer
    model_args.mm_vision_select_feature = getattr(
        model_config, "mm_vision_select_feature", "patch"
    )
    model_args.mm_patch_merge_type = getattr(
        model_config, "mm_patch_merge_type", "flat"
    )
    model_args.mm_projector_type = model_config.mm_projector_type
    model_args.pretrain_mm_mlp_adapter = f"{config.MODEL_PATH}/mm_projector.bin"

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    model = model.cuda()
    model.eval()

    # Convert projector to half precision
    for p in model.get_model().mm_projector.parameters():
        p.data = p.data.half()

    print(" Model loaded and ready for evaluation\n")
    return model, tokenizer


################################################################################
# DATASET LOADING
################################################################################
def load_dataset(config, tokenizer):
    """Load test dataset"""
    data_args = DataArgs(config)
    test_dataset = MotionLazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        data_split=config.DATA_SPLIT,
        num_frames=config.NUM_FRAMES,
        num_points=config.NUM_POINTS,
        vqa_path=data_args.vqa_path,
        motion_path=data_args.motion_path,
    )
    print(f"Loaded {len(test_dataset)} test samples\n")
    return test_dataset


################################################################################
# EVALUATION
################################################################################
def evaluate_sample(model, tokenizer, motion_tensor, question, config):
    """Generate prediction for a single question"""
    # Build prompt
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[config.CONVERSATION_VERSION].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Generate prediction
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=motion_tensor,
            do_sample=config.DO_SAMPLE,
            max_new_tokens=config.MAX_NEW_TOKENS,
            use_cache=config.USE_CACHE,
        )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return prediction


def main():
    config = Config()

    # Load model and dataset
    model, tokenizer = load_model(config)
    test_dataset = load_dataset(config, tokenizer)

    # Prepare output file
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.MODEL_PATH.split("/")[-1]
    output_file = f"{config.OUTPUT_DIR}/{model_name}_evaluation_{timestamp}.json"
    print(f"Results will be saved to: {output_file}\n")

    results = []

    # Process all samples
    for sample_idx in tqdm(
        range(len(test_dataset.list_data_dict)), desc="Evaluating samples"
    ):
        sample = test_dataset.list_data_dict[sample_idx]

        # Load motion sequence
        motion_info = {
            "frames": sample["frames"],
            "total_frames": sample["total_frames"],
        }
        motion_tensor = test_dataset._load_motion_sequence(motion_info)
        motion_tensor = motion_tensor.unsqueeze(0).half().cuda()

        # Process all QA pairs for this sample
        for qa_idx, qa_pair in enumerate(sample["qa_pairs"]):
            question = qa_pair["question"]
            answer = qa_pair["answer"]

            # Generate prediction
            prediction = evaluate_sample(
                model, tokenizer, motion_tensor, question, config
            )

            # Store result
            result = {
                "sample_idx": sample_idx,
                "qa_idx": qa_idx,
                "motion_id": sample.get("id", f"sample_{sample_idx}"),
                "question": question,
                "ground_truth": answer,
                "prediction": prediction,
                "total_frames": sample["total_frames"],
            }
            results.append(result)

            # Write to file in real-time
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

        # Clear CUDA cache periodically
        if (sample_idx + 1) % config.CACHE_CLEAR_INTERVAL == 0:
            torch.cuda.empty_cache()

    # Print summary
    total = len(results)
    print(f"\n Evaluation complete!")
    print(
        f" Processed {total} QA pairs from {len(test_dataset.list_data_dict)} samples"
    )
    print(f" Results saved to: {output_file}")


if __name__ == "__main__":
    main()

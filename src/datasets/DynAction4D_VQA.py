
import os
import json
import copy
import logging
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset


from llava.train.train import preprocess_multimodal, preprocess

logger = logging.getLogger(__name__)


class MotionLazySupervisedDataset(Dataset):
    """
    Loads motion sequences from PCD files and formats them for LLaVA training.
    Compatible with LLaVA's training pipeline but uses motion data (4D) instead of images.
    """

    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 data_args,
                 data_split: str = 'train',
                 num_frames: int = 32,
                 num_points: int = 2048,
                 vqa_path: str = None,
                 motion_path: str = None,
                 seed: int = 42):
        """
        Args:
            data_path: Base path to data (used if vqa_path/motion_path not specified)
            tokenizer: LLaVA tokenizer
            data_args: Data arguments from LLaVA
            data_split: Split name ('train', 'test', 'val')
            num_frames: Number of frames to sample per sequence
            num_points: Number of points to sample per frame
            vqa_path: Path to VQA directory 
            motion_path: Path to motion directory 
            seed: Random seed
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_frames = num_frames
        self.num_points = num_points
        self.stride = 2  # Fixed temporal sampling stride
        self.seed = seed
        
        self.categories = ['Action', 'Body-Spatial', 'Temporal']
        
        
        self.vqa_path = Path(vqa_path)
            
        self.motion_path = Path(motion_path)
        
        self.split = data_split
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        
        logger.info(f"Building motion dataset:")
        logger.info(f"  VQA path: {self.vqa_path}")
        logger.info(f"  Motion path: {self.motion_path}")
        logger.info(f"  Split: {self.split}")
        
        
        if not self.vqa_path.exists():
            raise ValueError(f"VQA path does not exist: {self.vqa_path}")
        if not self.motion_path.exists():
            raise ValueError(f"Motion path does not exist: {self.motion_path}")
        
        
        self.list_data_dict = self._discover_and_match_sequences()
        self.use_json_format = False
        
        if len(self.list_data_dict) == 0:
            raise ValueError(f"No samples found! Dataset is empty. Check your data path: {data_path}")
        
        logger.info(f"✓ Motion dataset initialized with {len(self.list_data_dict)} samples")

    def _discover_and_match_sequences(self) -> List[Dict]:
        """
        Scans VQA directory for sequences with *_vqa_pairs.json files,
        matches them with corresponding motion sequences,
        and converts to LLaVA format.
        """
        vqa_split_dir = self.vqa_path / self.split
        motion_split_dir = self.motion_path / self.split
        
        
        vqa_files = sorted(list(vqa_split_dir.rglob('*_vqa_pairs.json')))
        
        if len(vqa_files) == 0:
            vqa_files = sorted(list(vqa_split_dir.rglob('*_qna.json')))
        
        
        logger.info(f"Found {len(vqa_files)} VQA files in {vqa_split_dir}")
        print(f"Found {len(vqa_files)} VQA files in {vqa_split_dir}")
        
        samples = []
        matched_count = 0
        missing_motion = []
        
        for vqa_path in vqa_files:
            
            seq_name = vqa_path.parent.name
            
            
            motion_seq_dir = motion_split_dir / seq_name
            
            if not motion_seq_dir.exists():
                missing_motion.append(seq_name)
                continue
            
            
            frame_files = sorted(list(motion_seq_dir.glob('frame_*.pcd')))
            if not frame_files:
                logger.warning(f"No frames found for {seq_name}")
                continue
            
            
            frames = []
            for f in frame_files:
                try:
                    num = int(f.stem.split('_')[1])
                    frames.append((num, str(f)))
                except (IndexError, ValueError):
                    pass
            
            frames.sort(key=lambda x: x[0])
            
            if not frames:
                logger.warning(f"No valid frames for {seq_name}")
                continue
            
            
            try:
                with open(vqa_path, 'r') as f:
                    vqa_data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load VQA from {vqa_path}: {e}")
                continue
            
            
            if isinstance(vqa_data, dict) and 'qa_pairs' in vqa_data:
                vqa_data = vqa_data['qa_pairs']
            
            
            if not isinstance(vqa_data, list):
                logger.warning(f"VQA data is not a list in {vqa_path}")
                continue
            
            
            valid_qa_pairs = []
            for qa_pair in vqa_data:
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                if question and answer:
                    valid_qa_pairs.append(qa_pair)
            
            if not valid_qa_pairs:
                logger.warning(f"No valid Q&A pairs in {vqa_path}")
                continue
            
            
            sample = {
                "id": seq_name,
                "motion": str(motion_seq_dir),  
                "qa_pairs": valid_qa_pairs,  
                "frames": frames,
                "total_frames": len(frames),
            }
            samples.append(sample)
            matched_count += 1
        
        
        total_qa_pairs = sum(len(s['qa_pairs']) for s in samples)
        logger.info(f"Matched {matched_count}/{len(vqa_files)} sequences")
        logger.info(f"Created {len(samples)} sequences with {total_qa_pairs} total Q&A pairs")
        
        if missing_motion:
            logger.warning(f"Missing motion data for {len(missing_motion)} sequences (first 10): {missing_motion[:10]}")
        
        if len(samples) == 0:
            raise ValueError(
                f"No valid samples found!\n"
                f"VQA path: {vqa_split_dir}\n"
                f"Motion path: {motion_split_dir}\n"
                f"Found {len(vqa_files)} VQA files but no matches with motion data."
            )
        
        print(f"Matched {matched_count} sequences with {total_qa_pairs} Q&A pairs (train: random selection, test: first Q&A)")
        return samples

    def _load_pcd(self, pcd_path: str) -> np.ndarray:
        """
        Pure Python PCD loader to avoid Open3D segfaults.
        Returns (N, 9) array: [x, y, z, r, g, b, nx, ny, nz]
        """
        points, colors, normals = [], [], []
        
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA ascii'):
                    header_end = i + 1
                    break
            
            
            default_color = [0.5, 0.5, 0.5] 
            default_normal = [0.0, 0.0, 0.0]

            for line in lines[header_end:]:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # Parse XYZ
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                
                # Parse RGB
                c = default_color
                if len(parts) >= 4:
                    try:
                        
                        packed_rgb = int(float(parts[3]))
                        r = ((packed_rgb >> 16) & 255) / 255.0
                        g = ((packed_rgb >> 8) & 255) / 255.0
                        b = (packed_rgb & 255) / 255.0
                        c = [r, g, b]
                    except:
                        pass
                colors.append(c)
                
                # Parse Normals
                n = default_normal
                if len(parts) >= 7:
                    try:
                        n = [float(parts[4]), float(parts[5]), float(parts[6])]
                    except:
                        pass
                normals.append(n)

            # Convert to numpy
            pts = np.array(points, dtype=np.float32)
            cols = np.array(colors, dtype=np.float32)
            nrms = np.array(normals, dtype=np.float32)
            
            if len(pts) == 0: 
                return np.zeros((0, 9), dtype=np.float32)
            
            return np.concatenate([pts, cols, nrms], axis=1)

        except Exception as e:
            return np.zeros((0, 9), dtype=np.float32)

    def _process_points(self, point_data: np.ndarray) -> np.ndarray:
        """
        Samples or Pads points to self.num_points.
        Input: (N, 9). Output: (num_points, 9)
        """
        N = point_data.shape[0]
        T = self.num_points
        
        if N == 0:
            return np.zeros((T, 9), dtype=np.float32)
        
        if N >= T:
            indices = np.random.choice(N, T, replace=False)
            return point_data[indices]
        else:
            # Pad with zeros
            padding = np.zeros((T - N, 9), dtype=np.float32)
            return np.concatenate([point_data, padding], axis=0)

    def _sample_window_indices(self, total_frames: int) -> List[int]:
        """
        Implements Windowed Augmentation.
        - Training: Random window start position
        - Test/Val: Center window start position
        - Uses fixed stride=2 for temporal sampling
        - Clips that are too short are padded by clamping to last frame
        """
        # Calculate the total span required: (num_frames - 1) * stride + 1
        window_span = (self.num_frames - 1) * self.stride + 1
        
        if self.split == 'train':
            # RANDOM START for training
            if total_frames > window_span:
                start_frame = random.randint(0, total_frames - window_span)
            else:
                
                start_frame = 0
        else:
            # CENTERED START for test/val
            if total_frames > window_span:
                start_frame = (total_frames - window_span) // 2
            else:
                
                start_frame = 0

        # Sample frames with stride
        indices = []
        for i in range(self.num_frames):
            idx = start_frame + (i * self.stride)
            
            indices.append(min(idx, total_frames - 1))
        return indices

    def _augment_rotation(self, seq_data: np.ndarray) -> np.ndarray:
        """
        Rotates XYZ and Normals around Y axis by 0, 90, 180, or 270 degrees.
        seq_data: (T, N, 9) -> [x,y,z, r,g,b, nx,ny,nz]
        """
        k = random.choice([0, 1, 2, 3])
        if k == 0: return seq_data

        theta = k * (math.pi / 2)
        c, s = math.cos(theta), math.sin(theta)
        rot_mat = np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]], dtype=np.float32)

        # Apply to XYZ (indices 0,1,2)
        xyz = seq_data[:, :, 0:3]
        shape_orig = xyz.shape
        # Reshape to (Total_Points, 3) for matmul
        seq_data[:, :, 0:3] = (xyz.reshape(-1, 3) @ rot_mat.T).reshape(shape_orig)

        # Apply to Normals (indices 6,7,8)
        nrm = seq_data[:, :, 6:9]
        seq_data[:, :, 6:9] = (nrm.reshape(-1, 3) @ rot_mat.T).reshape(shape_orig)

        return seq_data

    def _load_motion_sequence(self, motion_info) -> torch.Tensor:
        """
        Load and process motion sequence.
        Returns: (T, N, 9) tensor
        """
        
        all_frames = motion_info['frames']
        
        # Sample frame indices
        indices = self._sample_window_indices(motion_info['total_frames'])
        selected_frames = [all_frames[i] for i in indices]
        
        
        seq_frames_processed = [] 
        
        for _, file_path in selected_frames:
            raw_data = self._load_pcd(file_path)
            processed_data = self._process_points(raw_data)
            seq_frames_processed.append(processed_data)
        
        # Stack to (T, N, 9) numpy array
        seq_data_np = np.stack(seq_frames_processed, axis=0)

        # Rotation Augmentation
        if self.split == 'train':
            seq_data_np = self._augment_rotation(seq_data_np)

        # 5. Global Normalization
        # Normalize coords (0:3) using min/max of the *entire sequence*
        xyz = seq_data_np[:, :, 0:3]
        min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
        max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0 # Prevent div/0
        
        seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
        # Convert to tensor - return as [T, N, 9]
        seq_data_tensor = torch.from_numpy(seq_data_np)  # [T, N, 9]
        
        return seq_data_tensor

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            motion_tokens = 197 if 'motion' in sample else 0  # Approx tokens from CL4D
            
            if 'qa_pairs' in sample:
                qa_pair = sample['qa_pairs'][0]
                text_len = len(qa_pair.get('question', '').split()) + len(qa_pair.get('answer', '').split())
            elif 'conversations' in sample:
                text_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            else:
                text_len = 0
            length_list.append(text_len + motion_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
        
            if 'qa_pairs' in sample:
                qa_pair = sample['qa_pairs'][0]
                cur_len = len(qa_pair.get('question', '').split()) + len(qa_pair.get('answer', '').split())
            elif 'conversations' in sample:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            else:
                cur_len = 0
            cur_len = cur_len if 'motion' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample in LLaVA format.
        
        For training: randomly selects one Q&A pair per sequence
        For testing: always selects the first Q&A pair for consistency
        """
        sample = self.list_data_dict[i]
        
        
        if 'qa_pairs' in sample:
            qa_pairs = sample['qa_pairs']
            if self.split == 'train':
                
                qa_pair = random.choice(qa_pairs)
            else:
                
                qa_pair = qa_pairs[0]
            
            
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            conversations = [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            
            sample = copy.deepcopy(sample)
            sample['conversations'] = conversations
        
        sources = [sample]
        
        # Load motion sequence if present
        if 'motion' in sources[0]:
            motion_path = self.list_data_dict[i]['motion']
            
            # Load motion data
            if 'frames' in self.list_data_dict[i]:
                # Use frame info from discovery
                motion_info = {
                    'frames': self.list_data_dict[i]['frames'],
                    'total_frames': self.list_data_dict[i]['total_frames']
                }
            else:
                
                motion_info = motion_path
            
            motion_sequence = self._load_motion_sequence(motion_info)
            
            # Process conversations
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # Tokenize conversations
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('motion' in self.list_data_dict[i]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])
        
        # Add motion data (renamed to 'image' for LLaVA compatibility)
        if 'motion' in self.list_data_dict[i]:
            data_dict['image'] = motion_sequence  # Shape: (T, N, 9)
        elif self.data_args.is_multimodal:
            # Dummy motion data if needed
            data_dict['image'] = torch.zeros(self.num_frames, self.num_points, 9)
        
        return data_dict


def make_motion_supervised_data_module(tokenizer, data_args) -> Dict:
    """
    Create motion-based dataset and collator for LLaVA training.
    
    This replaces make_supervised_data_module for motion-based training.
    
    Args:
        tokenizer: LLaVA tokenizer
        data_args: Should have:
            - data_path: Base directory containing VQA and motion subdirectories
            - vqa_path (optional): Explicit path to VQA directory
            - motion_path (optional): Explicit path to motion directory
    """
    from llava.train.train import DataCollatorForSupervisedDataset
    
    # Get VQA and motion paths from data_args if available
    vqa_path = getattr(data_args, 'vqa_path', None)
    motion_path = getattr(data_args, 'motion_path', None)
    
    train_dataset = MotionLazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        num_frames=32,
        num_points=2048,
        data_split='train',
        vqa_path=vqa_path,
        motion_path=motion_path,
    )
    
    
    eval_dataset = MotionLazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        num_frames=32,
        num_points=2048,
        data_split='test',
        vqa_path=vqa_path,
        motion_path=motion_path,
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

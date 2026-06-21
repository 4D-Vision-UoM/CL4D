import torch 
import numpy as np
import json
import logging
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class DynAction4DObj(torch.utils.data.Dataset):
    """
    DynAction4DObj dataset for Object Motion Sequences with Windowed Augmentation.
    """

    def __init__(self, 
                 data_root: str,
                 split: str = "train",
                 num_frames: int = 32,
                 num_points: int = 2048,
                 augment: bool = True,
                 seed: int = 42):
        """
        Args:
            data_root: Root directory of the dataset.
            split: "train", "val", or "test".
            num_frames: Number of frames to sample per sequence.
            num_points: Number of points to sample/pad per frame.
            augment: If True, apply rotation/text augmentations to 'train' split.
            seed: Random seed.
        """
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.num_frames = int(num_frames)
        self.num_points = int(num_points)
        self.stride = 2  # Fixed temporal interval
        
        # Enable augmentation only for train split if requested
        self.augment = (self.split == 'train') and augment
        self.seed = seed
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Data Discovery
        logger.info(f"Building dataset index for split='{split}' from root: {data_root}")
        self.samples = self._discover_samples()
        logger.info(f"Dataset initialized with {len(self.samples)} samples.")

    def _discover_samples(self) -> List[Dict]:
        """
        Finds all folders in the split directory 
        and indexes them as samples.
        """
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []
        
        
        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            
            frame_files = sorted(list(folder.glob('frame_*.pcd')))
            if not frame_files:
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
                continue

            
            sequence_name = folder.name

            samples.append({
                'folder_path': str(folder),
                'frames': frames, # List of (frame_num, file_path)
                'total_frames': len(frames),
                'sequence_name': sequence_name
            })
            
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
                
                # Parse Normals (columns 3,4,5)
                n = default_normal
                if len(parts) >= 6:
                    try:
                        n = [float(parts[3]), float(parts[4]), float(parts[5])]
                    except:
                        pass
                normals.append(n)
                
                
                c = default_color
                if len(parts) >= 7:
                    try:
                        
                        packed_rgb = int(float(parts[6]))
                        r = ((packed_rgb >> 16) & 255) / 255.0
                        g = ((packed_rgb >> 8) & 255) / 255.0
                        b = (packed_rgb & 255) / 255.0
                        c = [r, g, b]
                    except:
                        pass
                colors.append(c)

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns one full action segment with windowed sampling.
        """
        sample_info = self.samples[idx]
        all_frames = sample_info['frames']
        
        # 1. Text Augmentation
        # Extract caption from sequence_name (remove number at end and underscores)
        caption = ""
        try:
            sequence_name = sample_info['sequence_name']
            
            if '-' in sequence_name:
                caption = sequence_name.rsplit('-', 1)[0]
            else:
                caption = sequence_name
            
            caption = caption.replace('_', ' ')
        except Exception:
            pass  

        # 2. Frame Sampling (Windowed Augmentation)
        indices = self._sample_window_indices(sample_info['total_frames'])
        selected_frames = [all_frames[i] for i in indices]
        
        # 3. Load & Process PCDs
        seq_frames_processed = [] # Will hold (N, 9) arrays
        
        for _, file_path in selected_frames:
            raw_data = self._load_pcd(file_path)
            processed_data = self._process_points(raw_data)
            seq_frames_processed.append(processed_data)
        
        # Stack to (T, N, 9) numpy array
        seq_data_np = np.stack(seq_frames_processed, axis=0)

        # 4. Rotation Augmentation
        if self.augment:
            seq_data_np = self._augment_rotation(seq_data_np)

        # 5. Global Normalization
        # Normalize coords (0:3) using min/max of the *entire sequence*
        xyz = seq_data_np[:, :, 0:3]
        min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
        max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0 # Prevent div/0
        
        seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
        # 6. Convert to tensor - return as [T, N, 9]
        seq_data_tensor = torch.from_numpy(seq_data_np)  # [T, N, 9]

        return seq_data_tensor, caption

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    """
    Simple collate function to handle (tensor, string) tuples.
    """
    sequences = torch.stack([item[0] for item in batch], dim=0)  # [B, T, N, 9]
    captions = [item[1] for item in batch]  # List of strings
    return sequences, captions
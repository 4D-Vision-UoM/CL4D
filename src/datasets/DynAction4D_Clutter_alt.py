
import json
import logging
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DynAction4DClutter(torch.utils.data.Dataset):
    """
    DynAction4DClutter dataset for Cluttered Scenes with Windowed Augmentation.
    """

    def __init__(self, 
                 data_root: str,
                 split: str = "train",
                 num_frames: int = 32,
                 num_points: int = 2048,
                 no_normalize: bool = False,
                 augment: bool = True,
                 use_aug_scene: bool = False,
                 human_only: bool = False,
                 one_object_only: bool = False,
                 sphere_radius: float = 2.0,
                 seed: int = 42):
        """
        Args:
            data_root: Root directory of the dataset.
            split: "train", "val", or "test".
            num_frames: Number of frames to sample per sequence.
            num_points: Number of points to sample/pad per frame.
            grid_size: Voxel grid size (preserved in output).
            augment: If True, apply rotation/text augmentations to 'train' split.
            seed: Random seed.
        """
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.num_frames = int(num_frames)
        self.num_points = int(num_points)
        self.stride = 2  # Fixed temporal interval
        self.use_aug_scene = use_aug_scene
        self.human_only = human_only
        self.one_object_only = one_object_only
        self.augment = (self.split == 'train') and augment
        self.no_normalize = no_normalize
        self.seed = seed
        
        self.sphere_radius = sphere_radius
        
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
        Discovers sequences organized by environment.
        Structure: split/env_X/scene_point_cloud/scene.pcd
                   split/env_X/sequence_XXXXXX/frame_*.pcd
                   split/env_X/sequence_XXXXXX/report.json
        """
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []
        
        env_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('env_')])
        
        for env_dir in env_dirs:
            
            if self.use_aug_scene:
                scene_pcd_path = env_dir / 'scene_point_cloud' / 'aug.pcd'
            else:
                scene_pcd_path = env_dir / 'scene_point_cloud' / 'scene.pcd'
            if not scene_pcd_path.exists():
                logger.warning(f"Scene PCD not found for {env_dir.name}, skipping environment")
                continue
            
            
            sequence_dirs = sorted([d for d in env_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('sequence_')])
            
            for seq_dir in sequence_dirs:
                
                report_path = seq_dir / 'report.json'
                if not report_path.exists():
                    logger.warning(f"No report.json in {seq_dir}, skipping")
                    continue
                
                
                frame_files = sorted(list(seq_dir.glob('frame_*.pcd')))
                if not frame_files:
                    logger.warning(f"No frame files in {seq_dir}, skipping")
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

                samples.append({
                    'report_path': str(report_path),
                    'scene_pcd_path': str(scene_pcd_path),
                    'frames': frames,  
                    'total_frames': len(frames),
                    'sequence_name': seq_dir.name,
                    'environment': env_dir.name
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
                
                
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                
                
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
                
                
                n = default_normal
                if len(parts) >= 7:
                    try:
                        n = [float(parts[4]), float(parts[5]), float(parts[6])]
                    except:
                        pass
                normals.append(n)

            
            pts = np.array(points, dtype=np.float32)
            cols = np.array(colors, dtype=np.float32)
            nrms = np.array(normals, dtype=np.float32)
            
            if len(pts) == 0: 
                return np.zeros((0, 9), dtype=np.float32)
            
            return np.concatenate([pts, cols, nrms], axis=1)

        except Exception as e:
            
            return np.zeros((0, 9), dtype=np.float32)

    def _load_single_object_pcd(self, pcd_path: str) -> np.ndarray:
        """
        Reads a PCD, groups points by label, randomly selects ONE object, 
        and returns its points as an (N, 9) array: [x, y, z, r, g, b, nx, ny, nz]
        """
        all_objects = {}
        
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA ascii'):
                    header_end = i + 1
                    break
                    
            for line in lines[header_end:]:
                parts = line.strip().split()
                if len(parts) < 8: continue
                
                
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                nx, ny, nz = float(parts[3]), float(parts[4]), float(parts[5])
                
                # Decode packed RGB
                try:
                    packed_rgb = int(float(parts[6]))
                    r = ((packed_rgb >> 16) & 255) / 255.0
                    g = ((packed_rgb >> 8) & 255) / 255.0
                    b = (packed_rgb & 255) / 255.0
                except ValueError:
                    r, g, b = 0.5, 0.5, 0.5 # fallback
                    
                label = parts[7]
                
                if label not in all_objects:
                    all_objects[label] = []
                    
                all_objects[label].append([x, y, z, r, g, b, nx, ny, nz])
                
            if not all_objects:
                return np.zeros((0, 9), dtype=np.float32)
                
            
            valid_labels = [lbl for lbl in all_objects.keys() if "floor" not in lbl.lower() and "wall" not in lbl.lower()]
            
            
            if not valid_labels:
                valid_labels = list(all_objects.keys())
                
            
            chosen_label = random.choice(valid_labels)
            
            return np.array(all_objects[chosen_label], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to load single object from {pcd_path}: {e}")
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
            # Random subsample without replacement
            indices = np.random.choice(N, T, replace=False)
            return point_data[indices]
        else:
            # Pad with zeros
            padding = np.zeros((T - N, 9), dtype=np.float32)
            return np.concatenate([point_data, padding], axis=0)

    def _sample_window_indices(self, total_frames: int) -> List[int]:
        
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

    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
    #     """
    #     Returns one full action segment with windowed sampling.
    #     """
    #     sample_info = self.samples[idx]
    #     all_frames = sample_info['frames']
        
    #     # 1. Text Augmentation
        
    #     caption = ""
    #     try:
    #         with open(sample_info['report_path'], 'r') as f:
    #             meta = json.load(f)
    #             desc = meta.get('description', '')
    #             sentences = [s.strip() for s in desc.split('.') if s.strip()]
    #             if sentences:
    #                 # Randomly select one sentence if training and augmenting
    #                 caption = random.choice(sentences) if self.augment else sentences[0]
    #     except Exception:
    #         pass  

    #     # 2. Frame Sampling (Windowed Augmentation)
    #     indices = self._sample_window_indices(sample_info['total_frames'])
    #     selected_frames = [all_frames[i] for i in indices]
        
    #     # 3. Load & Process PCDs
    #     seq_frames_processed = []
        
    #     if self.one_object_only:
    #         scene_pcd = self._load_single_object_pcd(sample_info['scene_pcd_path'])
    #     else:
    #     #get the scene point cloud for this environment
    #         scene_pcd = self._load_pcd(sample_info['scene_pcd_path'])
        
            
        
    #     middle_frame_idx = len(selected_frames) // 2
    #     middle_frame_pcd = self._load_pcd(selected_frames[middle_frame_idx][1])
        
       
    #     human_center_xz = np.mean(middle_frame_pcd[:, [0, 2]], axis=0)
        
        
    #     scene_center_xz = np.mean(scene_pcd[:, [0, 2]], axis=0)
        
        
    #     shift_xz = human_center_xz - scene_center_xz
        
        
    #     scene_pcd[:, [0, 2]] += shift_xz
        
    #     for _, file_path in selected_frames:
    #         raw_data = self._load_pcd(file_path)
            
    #         if not self.human_only:
    #             raw_data = np.concatenate([raw_data, scene_pcd], axis=0)
    #         processed_data = self._process_points(raw_data)
    #         seq_frames_processed.append(processed_data)
        
        
    #     seq_data_np = np.stack(seq_frames_processed, axis=0)

    #     # Rotation Augmentation
    #     if self.augment:
    #         seq_data_np = self._augment_rotation(seq_data_np)
            
    #     if self.no_normalize:
    #         return torch.from_numpy(seq_data_np), caption

    #     # Global Normalization
    #     # Normalize coords (0:3) using min/max of the *entire sequence*
    #     xyz = seq_data_np[:, :, 0:3]
    #     min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
    #     max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
    #     range_xyz = max_xyz - min_xyz
    #     range_xyz[range_xyz == 0] = 1.0 # Prevent div/0
        
    #     seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
    #     # Convert to tensor - return as [T, N, 9]
    #     seq_data_tensor = torch.from_numpy(seq_data_np)  # [T, N, 9]

    #     return seq_data_tensor, caption

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample_info = self.samples[idx]
        all_frames = sample_info['frames']
        
        # 1. Text Augmentation logic... (omitted for brevity)
        caption = ""
        try:
            with open(sample_info['report_path'], 'r') as f:
                meta = json.load(f)
                desc = meta.get('description', '')
                sentences = [s.strip() for s in desc.split('.') if s.strip()]
                caption = random.choice(sentences) if self.augment and sentences else (sentences[0] if sentences else "")
        except Exception: pass

        # 2. Frame Sampling
        indices = self._sample_window_indices(sample_info['total_frames'])
        selected_frames = [all_frames[i] for i in indices]
        
        # 3. Load Static Scene
        if self.one_object_only:
            scene_pcd = self._load_single_object_pcd(sample_info['scene_pcd_path'])
        else:
            scene_pcd = self._load_pcd(sample_info['scene_pcd_path'])

        seq_frames_processed = []

        for _, file_path in selected_frames:
            # Step A: Load human frame
            human_pcd = self._load_pcd(file_path)
            
            if self.human_only:
                raw_combined = human_pcd
            else:
                # Step B: Get Human Centroid
                if human_pcd.shape[0] > 0:
                    human_centroid = np.mean(human_pcd[:, 0:3], axis=0)
                else:
                    human_centroid = np.array([0, 0, 0])

                # Step C: Combine Human and Scene first
                # (Assuming scene is already in the same coordinate space as human)
                raw_combined = np.concatenate([human_pcd, scene_pcd], axis=0)

                # Step D: Sphere Filtering on the COMBINED cloud
                # Calculate distance from human centroid for EVERY point (human + scene)
                dists = np.linalg.norm(raw_combined[:, 0:3] - human_centroid, axis=1)
                sphere_mask = dists <= self.sphere_radius
                
                # Keep only points within the sphere
                raw_combined = raw_combined[sphere_mask]

            # Step E: Usual processing (Subsample or Pad to num_points)
            processed_data = self._process_points(raw_combined)
            seq_frames_processed.append(processed_data)
        
        # Final sequence assembly and normalization...
        seq_data_np = np.stack(seq_frames_processed, axis=0)
        
        if self.augment:
            seq_data_np = self._augment_rotation(seq_data_np)
            
        if self.no_normalize:
            return torch.from_numpy(seq_data_np), caption

        # Global Normalization
        xyz = seq_data_np[:, :, 0:3]
        min_xyz, max_xyz = np.min(xyz, axis=(0, 1), keepdims=True), np.max(xyz, axis=(0, 1), keepdims=True)
        range_xyz = np.where((max_xyz - min_xyz) == 0, 1.0, max_xyz - min_xyz)
        seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
        return torch.from_numpy(seq_data_np), caption

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    """
    Simple collate function to handle (tensor, string) tuples.
    """
    sequences = torch.stack([item[0] for item in batch], dim=0)  # [B, T, N, 9]
    captions = [item[1] for item in batch]  # List of strings
    return sequences, captions








import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    # Check for explicit vision tower type specification (takes priority)
    vision_tower_type = getattr(vision_tower_cfg, "mm_vision_tower_type", None)

    # If explicitly specified as CL4D, use CL4D encoder
    if vision_tower_type and vision_tower_type.lower() == "cl4d":
        print(f"Building CL4D motion encoder : {vision_tower}")
        from .cl4d_encoder import CL4DVisionTower

        return CL4DVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower_type and vision_tower_type.lower() == "motionpointnet":
        print(f"Building MotionPointNet motion encoder: {vision_tower}")
        from .motion_pointnet import MotionPointNetVisionTower

        return MotionPointNetVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower_type and vision_tower_type.lower() == "psttransformer":
        print(f"Building PSTTransformer motion encoder: {vision_tower}")
        from .psttransformer import PSTTransformerVisionTower

        return PSTTransformerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower_type and vision_tower_type.lower() == "p4transformer":
        print(f"Building P4Transformer motion encoder: {vision_tower}")
        from .p4transformer import P4TransformerVisionTower

        return P4TransformerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Standard CLIP-based encoders
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")

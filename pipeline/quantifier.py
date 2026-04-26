import numpy as np
import config


def quantify_hemorrhage(mask: np.ndarray, spacing: tuple = None) -> dict:
    """Convert binary segmentation mask (2D or 3D) to hemorrhage volume in mL."""
    spacing = spacing or config.DEFAULT_SPACING
    voxel_ml = (spacing[0] / 10) * (spacing[1] / 10) * (spacing[2] / 10)  # mm³ → mL
    num_voxels = int(np.sum(mask > 0))
    volume_ml  = round(num_voxels * voxel_ml, 2)

    risk_level = config.get_risk_level(volume_ml)
    east_rec   = config.EAST_RECOMMENDATIONS.get(risk_level, "")

    return {
        "num_voxels":    num_voxels,
        "volume_ml":     volume_ml,
        "voxel_ml":      voxel_ml,
        "risk_level":    risk_level,
        "recommendation": east_rec,
    }

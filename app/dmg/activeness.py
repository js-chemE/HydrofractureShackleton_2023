import os
from pydantic import BaseModel, Field
import rioxarray as rioxr
import numpy as np

from typing import Any

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ActivenessResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    name: str | None = Field(default = None, description="Name of the tile")
    file_delta_alpha: str | None =  Field(default = None, description="Path to the delta alpha file")
    file_dmg: str | None = Field(default = None, description="Path to the damage file")

    activeness_angle: int = Field(default = 45, description="Angle to consider as active crevasses")
    activeness_spread: int = Field(default = 15, description="Spread of the angle to consider as active crevasses")

    angle_difference: Any | None = Field(default = None, description="Angle difference array")
    activeness_mask: Any | None = Field(default = None, description="Activeness mask array")
     

    def calculate(self):
        angles = rioxr.open_rasterio(self.file_delta_alpha)
        dmg = rioxr.open_rasterio(self.file_dmg)
        if angles.shape != dmg.shape:
            raise ValueError(f"Shape mismatch between delta-alpha {angles.shape} and damage {dmg.shape} files.")
        logger.info("Calculating activeness based:")
        logger.info(f" -> {self.file_delta_alpha}")
        logger.info(f" -> {self.file_dmg}")
        
        # Initialize the result arrays
        # angle_difference = np.zeros(angles.data[0].shape)
        self.activeness_mask = angles.copy()

        # Set 0 and dtypes
        # self.angle_difference.data = np.zeros(angles.shape, dtype=np.float32)
        self.activeness_mask.data = np.zeros(angles.shape, dtype=np.float32)

        # Calculate the active crevasses mask
        angle_difference = create_active_crevasses_mask(angles.data.squeeze(), self.activeness_angle, self.activeness_spread)
        self.activeness_mask.data[0, self.activeness_mask.data.squeeze() == 0] = np.nan
        self.activeness_mask.data[0, ~np.isfinite(angle_difference)] = np.nan
        self.activeness_mask.data[0, np.isfinite(angle_difference)] = 1

        # Mask out non-finite values and zero damage
        # self.activeness_mask.data[0, ~np.isfinite(dmg.data.squeeze())] = 0
        self.activeness_mask.data[0, dmg.data.squeeze() == 0] = 0

    def export(self, basename: str, path: str) -> None:
        """
        Export the results to a file.
        """
        new = basename + "_act.tif"
        logger.info(f"Exporting >{new}< to {path}")
        self.activeness_mask.rio.to_raster(raster_path=os.path.join(path, new))

def create_active_crevasses_mask(
    delta_alpha: np.ndarray, target: float = 45, spread: float = 15
):

    mask0 = np.abs(delta_alpha) <= 90  # only consider values between -90 and 90
    mask1 = np.abs(delta_alpha - target) <= spread  # target +- spread
    mask2 = np.abs(-delta_alpha - target) <= spread  # -target +- spread

    combined_mask1 = mask0 & mask1
    combined_mask2 = mask0 & mask2

    active_mask = np.zeros(delta_alpha.shape) * np.nan
    active_mask[combined_mask1] = 1
    active_mask[combined_mask2] = 1

    active = active_mask.copy()
    active[combined_mask1] = (
        delta_alpha[combined_mask1] * active_mask[combined_mask1] - target
    )
    active[combined_mask2] = (
        delta_alpha[combined_mask2] * active_mask[combined_mask2] + target
    )
    return active


def calculate_activeness(
        file_delta_alpha: str,
        file_dmg: str,
        target: float = 45,
        spread: float = 15
    ) -> ActivenessResult:
    
    result = ActivenessResult(
        file_delta_alpha=file_delta_alpha,
        file_dmg=file_dmg,
        activeness_angle=target,
        activeness_spread=spread
    )
    result.calculate()
    return result

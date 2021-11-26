import numpy as np

from typing import Sequence, Tuple

from simutome._simutome import Simutome


class VolumeSlicingEstimator:
    def __init__(
        self,
        data: np.ndarray,
        thickness: float,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        axes: Sequence[int] = (0, 1, 2),
    ) -> None:
        self.data = data
        self.thickness = thickness
        self.scale = scale
        self.axes = axes

    def estimate_cell_exclusion_probab(self) -> float:
        pass

    def estimate_cell_displacement_params(
        self,
    ) -> Simutome.CellDisplacementParams:
        pass

    def estimate_cell_split_params(self) -> Simutome.CellSplitParams:
        pass

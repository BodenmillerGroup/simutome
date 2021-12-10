import numpy as np
import pandas as pd

from itertools import product
from skimage.measure import regionprops
from tqdm.auto import tqdm
from typing import Generator, List, NamedTuple, Tuple


def measure_cell_mask(
    cell_mask: np.ndarray,
    voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pd.DataFrame:
    props_list = regionprops(cell_mask)
    return pd.DataFrame(
        data={
            "cell_id": [props.label for props in props_list],
            "cell_volume_um3": [
                p.area * np.prod(voxel_size_um) for p in props_list
            ],
            "cell_centroid_x_um": [
                props.centroid[-1] * voxel_size_um[-1] for props in props_list
            ],
            "cell_centroid_y_um": [
                props.centroid[-2] * voxel_size_um[-2] for props in props_list
            ],
            "cell_centroid_z_um": [
                props.centroid[-3] * voxel_size_um[-3] for props in props_list
            ],
            "proj_cell_area_x_um2": [
                np.sum(np.amax(props.image, axis=-1))
                * np.prod(voxel_size_um)
                / voxel_size_um[-1]
                for props in props_list
            ],
            "proj_cell_area_y_um2": [
                np.sum(np.amax(props.image, axis=-2))
                * np.prod(voxel_size_um)
                / voxel_size_um[-2]
                for props in props_list
            ],
            "proj_cell_area_z_um2": [
                np.sum(np.amax(props.image, axis=-3))
                * np.prod(voxel_size_um)
                / voxel_size_um[-3]
                for props in props_list
            ],
        }
    )


class CellMaskSlicer:
    class CellSliceInfo(NamedTuple):
        sectioning_axis: int
        section_thickness_um: float
        section_offset_um: float
        cell_id: int
        cell_slice_number: int
        cell_slice_volume_um3: float
        cell_slice_centroid_x_um: float
        cell_slice_centroid_y_um: float
        cell_slice_centroid_z_um: float
        proj_cell_slice_area_um2: float
        proj_cell_slice_centroid_x_um: float
        proj_cell_slice_centroid_y_um: float

    def __init__(
        self,
        cell_mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if cell_mask.ndim != 3:
            raise ValueError("mask")
        if len(voxel_size_um) != 3:
            raise ValueError("voxel_size_um")
        self._cell_mask = np.asarray(cell_mask)
        self._voxel_size_um = np.asarray(voxel_size_um)

    def get_section_count(
        self,
        sectioning_axis: int,
        section_thickness_um: float,
        section_offset_um: float = 0.0,
        drop_last: bool = True,
    ) -> int:
        section_count = (
            self._cell_mask.shape[sectioning_axis]
            * self.voxel_size_um[sectioning_axis]
            - section_offset_um
        ) / section_thickness_um
        if drop_last:
            section_count = np.floor(section_count)
        else:
            section_count = np.ceil(section_count)
        return int(section_count)

    def generate_sections(
        self,
        sectioning_axis: int,
        section_thickness_um: float,
        section_offset_um: float = 0.0,
        drop_last: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        if sectioning_axis not in (0, 1, 2):
            raise ValueError("sectioning_axis")
        if section_thickness_um <= 0.0 or np.any(
            (section_thickness_um % self._voxel_size_um) != 0.0
        ):
            raise ValueError("section_thickness_um")
        if section_offset_um < 0.0:
            raise ValueError("section_offset_um")
        pixel_size_um = self._voxel_size_um[sectioning_axis]
        cell_mask = np.moveaxis(self._cell_mask, sectioning_axis, 0)
        section_thickness_px = int(round(section_thickness_um / pixel_size_um))
        for section_start_um in np.arange(
            section_offset_um,
            cell_mask.shape[0] * pixel_size_um,
            section_thickness_um,
        ):
            section_start_px = int(round(section_start_um / pixel_size_um))
            section_stop_px = section_start_px + section_thickness_px
            if section_stop_px <= cell_mask.shape[0] or not drop_last:
                section = cell_mask[section_start_px:section_stop_px]
                yield np.moveaxis(section, 0, sectioning_axis)

    def run(
        self,
        sectioning_axes: List[int],
        section_thicknesses_um: List[float],
        drop_last: bool = True,
        progress: bool = False,
    ) -> List[CellSliceInfo]:
        if any(a not in (0, 1, 2) for a in sectioning_axes):
            raise ValueError("sectioning_axes")
        if any(
            t <= 0.0 or np.any((t % self._voxel_size_um) != 0.0)
            for t in section_thicknesses_um
        ):
            raise ValueError("section_thicknesses_um")
        cell_slice_infos = []
        voxel_volume_um3 = np.prod(self._voxel_size_um)
        if progress:
            progress_bar = tqdm(
                desc="Sections",
                total=sum(
                    self.get_section_count(
                        sectioning_axis,
                        section_thickness_um,
                        section_offset_um=section_offset_um,
                        drop_last=drop_last,
                    )
                    for sectioning_axis, section_thickness_um in product(
                        sectioning_axes, section_thicknesses_um
                    )
                    for section_offset_um in np.arange(
                        0.0, section_thickness_um, np.amax(self._voxel_size_um)
                    )
                ),
            )
        for sectioning_axis, section_thickness_um in product(
            sectioning_axes, section_thicknesses_um
        ):
            pixel_size_um = np.delete(self._voxel_size_um, sectioning_axis)
            pixel_area_um2 = np.prod(pixel_size_um)
            for section_offset_um in np.arange(
                0.0, section_thickness_um, np.amax(self._voxel_size_um)
            ):
                num_cell_slices = {}
                section_gen = self.generate_sections(
                    sectioning_axis,
                    section_thickness_um,
                    section_offset_um=section_offset_um,
                    drop_last=drop_last,
                )
                current_section_offset_um = section_offset_um
                for section in section_gen:
                    for props in regionprops(section):
                        cell_id = props.label
                        cell_slice_number = num_cell_slices.get(cell_id, 0)
                        cell_slice_volume_um3 = props.area * voxel_volume_um3
                        cell_slice_centroid_um = (
                            props.centroid * self._voxel_size_um
                        )
                        cell_slice_centroid_um[
                            sectioning_axis
                        ] += current_section_offset_um
                        proj = np.amax(props.image, axis=sectioning_axis)
                        proj_props = regionprops(proj.astype(np.uint8))[0]
                        proj_cell_slice_area_um2 = (
                            proj_props.area * pixel_area_um2
                        )
                        proj_cell_slice_centroid_um = (
                            proj_props.centroid * pixel_size_um
                        )
                        cell_slice_info = CellMaskSlicer.CellSliceInfo(
                            sectioning_axis,
                            section_thickness_um,
                            section_offset_um,
                            cell_id,
                            cell_slice_number,
                            cell_slice_volume_um3,
                            cell_slice_centroid_um[-1],
                            cell_slice_centroid_um[-2],
                            cell_slice_centroid_um[-3],
                            proj_cell_slice_area_um2,
                            proj_cell_slice_centroid_um[-1],
                            proj_cell_slice_centroid_um[-2],
                        )
                        cell_slice_infos.append(cell_slice_info)
                        num_cell_slices[cell_id] = cell_slice_number + 1
                    current_section_offset_um += section_thickness_um
                    if progress:
                        progress_bar.update()
        if progress:
            progress_bar.close()
        return cell_slice_infos

    @property
    def cell_mask(self) -> np.ndarray:
        return self._cell_mask

    @property
    def voxel_size_um(self) -> Tuple[float, float, float]:
        return tuple(self._voxel_size_um)

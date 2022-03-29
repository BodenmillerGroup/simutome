import itertools
from typing import Generator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage.measure import regionprops
from tqdm.auto import tqdm


class CellSlicer:
    def __init__(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        channel_names: Optional[Sequence[str]] = None,
        voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if mask.ndim != 3:
            raise ValueError("mask")
        if image is not None and (
            image.ndim != 4
            or image.shape[0] != mask.shape[0]
            or image.shape[2:] != mask.shape[1:]
        ):
            raise ValueError("image")
        if image is not None and (
            channel_names is None or len(channel_names) != image.shape[1]
        ):
            raise ValueError("channel_names")
        if len(voxel_size_um) != 3:
            raise ValueError("voxel_size_um")
        self._mask = mask
        self._image = image
        self._channel_names = channel_names
        self._voxel_size_um = np.asarray(voxel_size_um)

    def get_section_count(
        self,
        sectioning_axis: int,
        section_thickness_um: float,
        section_offset_um: float = 0.0,
        drop_last: bool = True,
    ) -> int:
        section_count = (
            self._mask.shape[sectioning_axis] * self.voxel_size_um[sectioning_axis]
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
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        if sectioning_axis not in (0, 1, 2):
            raise ValueError("sectioning_axis")
        if section_thickness_um <= 0.0 or np.any(
            (section_thickness_um % self._voxel_size_um) != 0.0
        ):
            raise ValueError("section_thickness_um")
        if section_offset_um < 0.0:
            raise ValueError("section_offset_um")
        mask = np.moveaxis(self._mask, sectioning_axis, 0)
        if self._image is not None:
            img = np.moveaxis(self._image, sectioning_axis + (sectioning_axis > 0), 0)
        else:
            img = None
        pixel_size_um = self._voxel_size_um[sectioning_axis]
        section_thickness_px = int(round(section_thickness_um / pixel_size_um))
        for section_start_um in np.arange(
            section_offset_um,
            mask.shape[0] * pixel_size_um,
            section_thickness_um,
        ):
            section_start_px = int(round(section_start_um / pixel_size_um))
            section_stop_px = section_start_px + section_thickness_px
            if section_stop_px <= mask.shape[0] or not drop_last:
                mask_section = np.moveaxis(
                    mask[section_start_px:section_stop_px], 0, sectioning_axis
                )
                if img is not None:
                    img_section = np.moveaxis(
                        img[section_start_px:section_stop_px],
                        0,
                        sectioning_axis + (sectioning_axis > 0),
                    )
                else:
                    img_section = None
                yield mask_section, img_section

    def measure_cells(
        self,
        sectioning_axes: Sequence[int],
        progress: bool = False,
    ) -> pd.DataFrame:
        if any(a not in (0, 1, 2) for a in sectioning_axes):
            raise ValueError("sectioning_axes")
        cell_info_data = []
        voxel_volume_um3 = np.prod(self._voxel_size_um)
        cell_props_list = regionprops(
            self._mask,
            intensity_image=(
                np.moveaxis(self._image, 1, -1) if self._image is not None else None
            ),
        )
        if progress:
            pbar = tqdm(total=len(sectioning_axes) * len(cell_props_list))
        for sectioning_axis in sectioning_axes:
            pixel_size_um = np.delete(self._voxel_size_um, sectioning_axis)
            pixel_area_um2 = np.prod(pixel_size_um)
            for cell_props in cell_props_list:
                cell_id = cell_props.label
                cell_volume_um3 = cell_props.area * voxel_volume_um3
                cell_centroid_um = cell_props.centroid * np.array(self._voxel_size_um)
                proj_cell_props = regionprops(
                    np.amax(cell_props.image, axis=sectioning_axis).astype(np.uint8),
                    intensity_image=np.sum(
                        cell_props.image_intensity, axis=sectioning_axis
                    ),
                )[0]
                proj_cell_area_um2 = proj_cell_props.area * pixel_area_um2
                proj_cell_centroid_um = list(proj_cell_props.centroid * pixel_size_um)
                proj_cell_centroid_um.insert(sectioning_axis, float("nan"))
                cell_info_row = [
                    sectioning_axis,
                    cell_id,
                    cell_volume_um3,
                    cell_centroid_um[-1],
                    cell_centroid_um[-2],
                    cell_centroid_um[-3],
                    proj_cell_area_um2,
                    proj_cell_centroid_um[-1],
                    proj_cell_centroid_um[-2],
                    proj_cell_centroid_um[-3],
                ]
                if self._image is not None:
                    cell_info_row += cell_props.intensity_mean.tolist()
                    cell_info_row += proj_cell_props.intensity_mean.tolist()
                cell_info_data.append(cell_info_row)
                if progress:
                    pbar.update()
        if progress:
            pbar.close()
        cell_info_columns = [
            "sectioning_axis",
            "cell_id",
            "cell_volume_um3",
            "cell_centroid_x_um",
            "cell_centroid_y_um",
            "cell_centroid_z_um",
            "proj_cell_area_um2",
            "proj_cell_centroid_x_um",
            "proj_cell_centroid_y_um",
            "proj_cell_centroid_z_um",
        ]
        if self._image is not None:
            cell_info_columns += [
                f"mean_cell_intensity_{channel_name}"
                for channel_name in self._channel_names
            ]
            cell_info_columns += [
                f"mean_proj_cell_intensity_{channel_name}"
                for channel_name in self._channel_names
            ]
        return pd.DataFrame(data=cell_info_data, columns=cell_info_columns, copy=False)

    def measure_cell_slices(
        self,
        sectioning_axes: Sequence[int],
        section_thicknesses_um: Sequence[float],
        drop_last: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        if any(a not in (0, 1, 2) for a in sectioning_axes):
            raise ValueError("sectioning_axes")
        if any(
            t <= 0.0 or np.any((t % self._voxel_size_um) != 0.0)
            for t in section_thicknesses_um
        ):
            raise ValueError("section_thicknesses_um")
        cell_slice_info_data = []
        voxel_volume_um3 = np.prod(self._voxel_size_um)
        if progress:
            pbar = tqdm(
                total=sum(
                    self.get_section_count(
                        sectioning_axis,
                        section_thickness_um,
                        section_offset_um=section_offset_um,
                        drop_last=drop_last,
                    )
                    for sectioning_axis in sectioning_axes
                    for section_thickness_um in section_thicknesses_um
                    for section_offset_um in np.arange(
                        0.0, section_thickness_um, np.amax(self._voxel_size_um)
                    )
                ),
            )
        for sectioning_axis, section_thickness_um in itertools.product(
            sectioning_axes, section_thicknesses_um
        ):
            pixel_size_um = np.delete(self._voxel_size_um, sectioning_axis)
            pixel_area_um2 = np.prod(pixel_size_um)
            for section_offset_um in np.arange(
                0.0, section_thickness_um, np.amax(self._voxel_size_um)
            ):
                num_cell_slices = {}
                current_section_offset_um = section_offset_um
                for mask_section, img_section in self.generate_sections(
                    sectioning_axis,
                    section_thickness_um,
                    section_offset_um=section_offset_um,
                    drop_last=drop_last,
                ):
                    for cell_slice_props in regionprops(
                        mask_section,
                        intensity_image=(
                            np.moveaxis(img_section, 1, -1)
                            if img_section is not None
                            else None
                        ),
                    ):
                        cell_id = cell_slice_props.label
                        cell_slice_number = num_cell_slices.get(cell_id, 0)
                        cell_slice_volume_um3 = cell_slice_props.area * voxel_volume_um3
                        cell_slice_centroid_um = list(
                            cell_slice_props.centroid * self._voxel_size_um
                        )
                        cell_slice_centroid_um[
                            sectioning_axis
                        ] += current_section_offset_um
                        proj_cell_slice_props = regionprops(
                            np.amax(
                                cell_slice_props.image, axis=sectioning_axis
                            ).astype(np.uint8),
                            intensity_image=np.sum(
                                cell_slice_props.image_intensity,
                                axis=sectioning_axis,
                            ),
                        )[0]
                        proj_cell_slice_area_um2 = (
                            proj_cell_slice_props.area * pixel_area_um2
                        )
                        proj_cell_slice_centroid_um = list(
                            proj_cell_slice_props.centroid * pixel_size_um
                        )
                        proj_cell_slice_centroid_um.insert(
                            sectioning_axis, float("nan")
                        )
                        cell_slice_info_row = [
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
                            proj_cell_slice_centroid_um[-3],
                        ]
                        if self._image is not None:
                            cell_slice_info_row += (
                                cell_slice_props.intensity_mean.tolist()
                            )
                            cell_slice_info_row += (
                                proj_cell_slice_props.intensity_mean.tolist()
                            )
                        cell_slice_info_data.append(cell_slice_info_row)
                        num_cell_slices[cell_id] = cell_slice_number + 1
                    current_section_offset_um += section_thickness_um
                    if progress:
                        pbar.update()
        if progress:
            pbar.close()
        cell_slice_info_columns = [
            "sectioning_axis",
            "section_thickness_um",
            "section_offset_um",
            "cell_id",
            "cell_slice_number",
            "cell_slice_volume_um3",
            "cell_slice_centroid_x_um",
            "cell_slice_centroid_y_um",
            "cell_slice_centroid_z_um",
            "proj_cell_slice_area_um2",
            "proj_cell_slice_centroid_x_um",
            "proj_cell_slice_centroid_y_um",
            "proj_cell_slice_centroid_z_um",
        ]
        if self._image is not None:
            cell_slice_info_columns += [
                f"mean_cell_slice_intensity_{channel_name}"
                for channel_name in self._channel_names
            ]
            cell_slice_info_columns += [
                f"mean_proj_cell_slice_intensity_{channel_name}"
                for channel_name in self._channel_names
            ]
        return pd.DataFrame(
            data=cell_slice_info_data, columns=cell_slice_info_columns, copy=False
        )

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def channel_names(self) -> Optional[Sequence[str]]:
        return self._channel_names

    @property
    def voxel_size_um(self) -> Tuple[float, float, float]:
        return tuple(self._voxel_size_um)

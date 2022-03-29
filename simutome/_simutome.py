from typing import Generator, Optional, Tuple

import numpy as np
from skimage.transform import AffineTransform


class Simutome:
    def __init__(
        self,
        image_occlusion: float = 0.0,
        image_scale: Tuple[float, float] = (1.0, 1.0),
        image_rotation: float = 0.0,
        image_shear: float = 0.0,
        image_translation: Tuple[float, float] = (0.0, 0.0),
        exclude_cells: bool = False,
        cell_radius_mean: Optional[float] = None,
        cell_radius_std: Optional[float] = None,
        displace_cells: bool = False,
        cell_displacement_mean: Optional[float] = None,
        cell_displacement_var: Optional[float] = None,
        cell_division_probab: float = 0.0,
        cell_division_dist_mean: Optional[float] = None,
        cell_division_dist_std: Optional[float] = None,
        cell_swapping_probab: float = 0.0,
        shuffle_cells: bool = True,
        seed=None,
    ) -> None:
        if image_occlusion < 0.0 or image_occlusion >= 1.0:
            raise ValueError("image_occlusion")
        if image_scale[0] <= 0.0 or image_scale[1] <= 0.0:
            raise ValueError("image_scale")
        if exclude_cells and cell_radius_mean is None:
            raise ValueError("cell_radius_mean")
        if exclude_cells and cell_radius_std is None:
            raise ValueError("cell_radius_std")
        if displace_cells and cell_displacement_mean is None:
            raise ValueError("cell_displacement_mean")
        if displace_cells and cell_displacement_var is None:
            raise ValueError("cell_displacement_var")
        if cell_division_probab < 0.0 or cell_division_probab > 1.0:
            raise ValueError("cell_division_probab")
        if cell_division_probab > 0.0 and cell_division_dist_mean is None:
            raise ValueError("cell_division_dist_mean")
        if cell_division_probab > 0.0 and cell_division_dist_std is None:
            raise ValueError("cell_division_dist_std")
        if cell_swapping_probab < 0.0 or cell_swapping_probab > 1.0:
            raise ValueError("cell_swapping_probab")
        self.image_occlusion = image_occlusion
        self.image_scale = image_scale
        self.image_rotation = image_rotation
        self.image_shear = image_shear
        self.image_translation = image_translation
        self.exclude_cells = exclude_cells
        self.cell_radius_mean = cell_radius_mean
        self.cell_radius_std = cell_radius_std
        self.displace_cells = displace_cells
        self.cell_displacement_mean = cell_displacement_mean
        self.cell_displacement_var = cell_displacement_var
        self.cell_division_probab = cell_division_probab
        self.cell_division_dist_mean = cell_division_dist_mean
        self.cell_division_dist_std = cell_division_dist_std
        self.cell_swapping_probab = cell_swapping_probab
        self.shuffle_cells = shuffle_cells
        self._rng = np.random.default_rng(seed=seed)

    def generate_sections(
        self,
        cell_coords: np.ndarray,
        section_thickness: float,
        image_size: Optional[Tuple[int, int]] = None,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        if cell_coords.ndim != 2 or cell_coords.shape[1] != 2:
            raise ValueError("cell_coords")
        if section_thickness <= 0.0:
            raise ValueError("section_thickness")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_coords.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_coords.shape[0]
        ):
            raise ValueError("cell_clusters")
        if n is not None and n <= 0:
            raise ValueError("n")
        i = 0
        while n is None or i < n:
            yield self.generate_section(
                cell_coords,
                section_thickness,
                image_size=image_size,
                cell_intensities=cell_intensities,
                cell_clusters=cell_clusters,
            )
            i += 1

    def generate_section(
        self,
        cell_coords: np.ndarray,
        section_thickness: float,
        image_size: Optional[Tuple[int, int]] = None,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if cell_coords.ndim != 2 or cell_coords.shape[1] != 2:
            raise ValueError("cell_coords")
        if section_thickness <= 0.0:
            raise ValueError("section_thickness")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_coords.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_coords.shape[0]
        ):
            raise ValueError("cell_clusters")
        cell_indices = np.arange(len(cell_coords))
        if self.image_occlusion > 0.0:
            if image_size is None:
                raise ValueError("image_size")
            image_occlusion_mask = self._occlude_image(cell_coords, image_size)
            cell_coords = cell_coords[~image_occlusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~image_occlusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~image_occlusion_mask]
            cell_indices = cell_indices[~image_occlusion_mask]
        if self.exclude_cells:
            cell_exclusion_mask = self._exclude_cells(
                len(cell_coords), section_thickness
            )
            cell_coords = cell_coords[~cell_exclusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~cell_exclusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~cell_exclusion_mask]
            cell_indices = cell_indices[~cell_exclusion_mask]
        if self.displace_cells:
            cell_coords += self._rng.multivariate_normal(
                np.ones(2) * self.cell_displacement_mean,
                np.eye(2) * self.cell_displacement_var,
                size=len(cell_coords),
            )
        if self.cell_swapping_probab > 0.0:
            if cell_intensities is None:
                raise ValueError("cell_intensities")
            if cell_clusters is None:
                raise ValueError("cell_clusters")
            cell_swapping_indices = self._swap_cells(cell_intensities, cell_clusters)
            cell_intensities = cell_intensities[cell_swapping_indices]
        if self.cell_division_probab > 0.0:
            cell_coords, cell_division_indices = self._divide_cells(cell_coords)
            if cell_intensities is not None:
                cell_intensities = cell_intensities[cell_division_indices]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[cell_division_indices]
            cell_indices = cell_indices[cell_division_indices]
        if (
            self.image_scale != (1.0, 1.0)
            or self.image_rotation != 0.0
            or self.image_shear != 0.0
            or self.image_translation != (0.0, 0.0)
        ):
            if image_size is None:
                raise ValueError("image_size")
            cell_coords = self._transform_image(cell_coords, image_size)
        if self.shuffle_cells:
            cell_shuffling_indices = self._rng.permutation(len(cell_coords))
            cell_coords = cell_coords[cell_shuffling_indices]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[cell_shuffling_indices]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[cell_shuffling_indices]
            cell_indices = cell_indices[cell_shuffling_indices]
        return cell_indices, cell_coords, cell_intensities

    def _occlude_image(
        self,
        cell_coords: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.array:
        w = int(round(image_size[0] * (1 - self.image_occlusion) ** 0.5))
        h = int(round(image_size[1] * (1 - self.image_occlusion) ** 0.5))
        x0 = self._rng.integers(image_size[0] - w)
        y0 = self._rng.integers(image_size[1] - h)
        return (
            (cell_coords[:, 0] < x0)
            & (cell_coords[:, 0] >= x0 + w)
            & (cell_coords[:, 1] < y0)
            & (cell_coords[:, 1] >= y0 + h)
        )

    def _transform_image(
        self, cell_coords: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        c = AffineTransform(translation=(-image_size[0] / 2.0, -image_size[1] / 2.0))
        t = AffineTransform(
            scale=self.image_scale,
            rotation=self.image_rotation,
            shear=self.image_shear,
            translation=self.image_translation,
        )
        return c.inverse(t(c(cell_coords)))

    def _exclude_cells(
        self, num_cells: int, section_thickness: float, k: float = 1e-12
    ) -> np.ndarray:
        d = self._rng.normal(
            loc=2.0 * self.cell_radius_mean,
            scale=2.0 * self.cell_radius_std,
            size=num_cells,
        )
        s_physical = np.ceil(d / section_thickness)
        n1 = np.floor((section_thickness - (d % section_thickness)) / k) + 1
        n2 = np.ceil((d % section_thickness) / k) - 1
        s_discrete = (n1 * s_physical + n2 * (s_physical + 1)) / (n1 + n2)
        return self._rng.random(size=num_cells) < 1.0 / s_discrete

    def _swap_cells(
        self, cell_data: np.ndarray, cell_clusters: np.ndarray
    ) -> np.ndarray:
        ind = np.arange(len(cell_data))
        n = self._rng.binomial(len(cell_data), self.cell_swapping_probab)
        if n == 0:
            return ind
        for i in self._rng.choice(len(cell_data), size=n, replace=False):
            m = cell_clusters == cell_clusters[i]
            j = self._rng.choice(len(cell_data), p=m / np.sum(m))
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def _divide_cells(self, cell_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self._rng.binomial(len(cell_coords), self.cell_division_probab)
        ind = self._rng.integers(len(cell_coords), size=n)
        radii = self._rng.uniform(low=0.0, high=np.pi, size=n)
        dists = self._rng.normal(
            loc=self.cell_division_dist_mean,
            scale=self.cell_division_dist_std,
            size=n,
        )
        deltas = dists[:, None] * np.column_stack((np.cos(radii), np.sin(radii)))
        cell_division_indices = np.concatenate(
            (np.delete(np.arange(len(cell_coords)), ind), ind, ind)
        )
        cell_coords = np.concatenate(
            (
                np.delete(cell_coords, ind, axis=0),
                cell_coords[ind] + deltas,
                cell_coords[ind] - deltas,
            )
        )
        return cell_coords, cell_division_indices

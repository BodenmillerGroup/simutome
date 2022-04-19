import sys
from typing import Generator, Optional

import numpy as np
from scipy.stats import truncnorm
from skimage.transform import AffineTransform


Section = tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


class Simutome:
    def __init__(
        self,
        image_occlusion: float = 0.0,
        image_scale: tuple[float, float] = (1.0, 1.0),
        image_rotation: float = 0.0,
        image_shear: float = 0.0,
        image_translation: tuple[float, float] = (0.0, 0.0),
        exclude_cells: bool = False,
        section_thickness: Optional[float] = None,
        cell_diameter_mean: Optional[float] = None,
        cell_diameter_std: Optional[float] = None,
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
        if exclude_cells and (section_thickness is None or section_thickness <= 0.0):
            raise ValueError("section_thickness")
        if exclude_cells and (cell_diameter_mean is None or cell_diameter_mean <= 0.0):
            raise ValueError("cell_diameter_mean")
        if exclude_cells and (cell_diameter_std is None or cell_diameter_std <= 0.0):
            raise ValueError("cell_diameter_std")
        if displace_cells and (
            cell_displacement_mean is None or cell_displacement_mean <= 0.0
        ):
            raise ValueError("cell_displacement_mean")
        if displace_cells and (
            cell_displacement_var is None or cell_displacement_var <= 0.0
        ):
            raise ValueError("cell_displacement_var")
        if cell_division_probab < 0.0 or cell_division_probab > 1.0:
            raise ValueError("cell_division_probab")
        if cell_division_probab > 0.0 and (
            cell_division_dist_mean is None or cell_division_dist_mean <= 0.0
        ):
            raise ValueError("cell_division_dist_mean")
        if cell_division_probab > 0.0 and (
            cell_division_dist_std is None or cell_division_dist_std <= 0.0
        ):
            raise ValueError("cell_division_dist_std")
        if cell_swapping_probab < 0.0 or cell_swapping_probab > 1.0:
            raise ValueError("cell_swapping_probab")
        self.image_occlusion = image_occlusion
        self.image_scale = image_scale
        self.image_rotation = image_rotation
        self.image_shear = image_shear
        self.image_translation = image_translation
        self.exclude_cells = exclude_cells
        self.section_thickness = section_thickness
        self.cell_diameter_mean = cell_diameter_mean
        self.cell_diameter_std = cell_diameter_std
        self.displace_cells = displace_cells
        self.cell_displacement_mean = cell_displacement_mean
        self.cell_displacement_var = cell_displacement_var
        self.cell_division_probab = cell_division_probab
        self.cell_division_dist_mean = cell_division_dist_mean
        self.cell_division_dist_std = cell_division_dist_std
        self.cell_swapping_probab = cell_swapping_probab
        self.shuffle_cells = shuffle_cells
        self._seed_rng = np.random.default_rng(seed=seed)

    def skip_sections(self, n: int) -> None:
        if n < 0:
            raise ValueError("n")
        self._seed_rng.integers(sys.maxsize, size=n)

    def skip_section_pairs(self, n: int) -> None:
        if n < 0:
            raise ValueError("n")
        self._seed_rng.integers(sys.maxsize, size=2 * n)

    def generate_sections(
        self,
        cell_points: np.ndarray,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
        n: Optional[int] = None,
    ) -> Generator[Section, None, None]:
        if cell_points.ndim != 2 or cell_points.shape[1] != 2:
            raise ValueError("cell_points")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_clusters")
        if n is not None and n <= 0:
            raise ValueError("n")
        i = 0
        while n is None or i < n:
            yield self.generate_section(
                cell_points,
                cell_intensities=cell_intensities,
                cell_clusters=cell_clusters,
                image_size=image_size,
            )
            i += 1

    def generate_section_pairs(
        self,
        cell_points: np.ndarray,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
        n: Optional[int] = None,
    ) -> Generator[tuple[Section, Section], None, None]:
        if cell_points.ndim != 2 or cell_points.shape[1] != 2:
            raise ValueError("cell_points")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_clusters")
        if n is not None and n <= 0:
            raise ValueError("n")
        i = 0
        while n is None or i < n:
            yield self.generate_section_pair(
                cell_points,
                cell_intensities=cell_intensities,
                cell_clusters=cell_clusters,
                image_size=image_size,
            )
            i += 1

    def generate_section(
        self,
        cell_points: np.ndarray,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> Section:
        if cell_points.ndim != 2 or cell_points.shape[1] != 2:
            raise ValueError("cell_points")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_clusters")
        section_rng = np.random.default_rng(seed=self._seed_rng.integers(sys.maxsize))
        cell_indices = np.arange(len(cell_points))
        if self.image_occlusion > 0.0:
            if image_size is None:
                raise ValueError("image_size")
            image_occlusion_mask = self._occlude_image(cell_points, image_size)
            cell_points = cell_points[~image_occlusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~image_occlusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~image_occlusion_mask]
            cell_indices = cell_indices[~image_occlusion_mask]
        if self.exclude_cells:
            cell_exclusion_mask = self._exclude_cells(len(cell_points), section_rng)
            cell_points = cell_points[~cell_exclusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~cell_exclusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~cell_exclusion_mask]
            cell_indices = cell_indices[~cell_exclusion_mask]
        if self.displace_cells:
            cell_points += section_rng.multivariate_normal(
                np.ones(2) * self.cell_displacement_mean,
                np.eye(2) * self.cell_displacement_var,
                size=len(cell_points),
            )
        if self.cell_swapping_probab > 0.0:
            if cell_intensities is None:
                raise ValueError("cell_intensities")
            if cell_clusters is None:
                raise ValueError("cell_clusters")
            cell_swapping_indices = self._swap_cells(
                cell_intensities, cell_clusters, section_rng
            )
            cell_intensities = cell_intensities[cell_swapping_indices]
        if self.cell_division_probab > 0.0:
            cell_points, cell_division_indices = self._divide_cells(
                cell_points, section_rng
            )
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
            cell_points = self._transform_image(cell_points)
        if self.shuffle_cells:
            cell_shuffling_indices = section_rng.permutation(len(cell_points))
            cell_points = cell_points[cell_shuffling_indices]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[cell_shuffling_indices]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[cell_shuffling_indices]
            cell_indices = cell_indices[cell_shuffling_indices]
        return cell_indices, cell_points, cell_intensities

    def generate_section_pair(
        self,
        cell_points: np.ndarray,
        cell_intensities: Optional[np.ndarray] = None,
        cell_clusters: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> tuple[Section, Section]:
        if cell_points.ndim != 2 or cell_points.shape[1] != 2:
            raise ValueError("cell_points")
        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size")
        if cell_intensities is not None and (
            cell_intensities.ndim != 2
            or cell_intensities.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_intensities")
        if cell_clusters is not None and (
            cell_clusters.ndim != 1 or cell_clusters.shape[0] != cell_points.shape[0]
        ):
            raise ValueError("cell_clusters")
        section_rng = np.random.default_rng(seed=self._seed_rng.integers(sys.maxsize))
        cell_points1 = cell_points
        cell_points2 = cell_points
        cell_intensities1 = cell_intensities
        cell_intensities2 = cell_intensities
        cell_clusters1 = cell_clusters
        cell_clusters2 = cell_clusters
        cell_indices1 = np.arange(len(cell_points1))
        cell_indices2 = np.arange(len(cell_points2))
        if self.image_occlusion > 0.0:
            if image_size is None:
                raise ValueError("image_size")
            image_occlusion_mask1 = self._occlude_image(
                cell_points1, image_size, first=True
            )
            image_occlusion_mask2 = self._occlude_image(
                cell_points2, image_size, first=False
            )
            cell_points1 = cell_points1[~image_occlusion_mask1]
            cell_points2 = cell_points2[~image_occlusion_mask2]
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[~image_occlusion_mask1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[~image_occlusion_mask2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[~image_occlusion_mask1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[~image_occlusion_mask2]
            cell_indices1 = cell_indices1[~image_occlusion_mask1]
            cell_indices2 = cell_indices2[~image_occlusion_mask2]
        if self.exclude_cells:
            cell_exclusion_mask1 = self._exclude_cells(
                len(cell_points1), section_rng, first=True
            )
            cell_exclusion_mask2 = self._exclude_cells(
                len(cell_points2), section_rng, first=False
            )
            cell_points1 = cell_points1[~cell_exclusion_mask1]
            cell_points2 = cell_points2[~cell_exclusion_mask2]
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[~cell_exclusion_mask1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[~cell_exclusion_mask2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[~cell_exclusion_mask1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[~cell_exclusion_mask2]
            cell_indices1 = cell_indices1[~cell_exclusion_mask1]
            cell_indices2 = cell_indices2[~cell_exclusion_mask2]
        if self.displace_cells:
            displacement = section_rng.multivariate_normal(
                0.5 * np.ones(2) * self.cell_displacement_mean,
                0.25 * np.eye(2) * self.cell_displacement_var,
                size=len(np.intersect1d(cell_indices1, cell_indices2)),
            )
            cell_points1[np.isin(cell_indices1, cell_indices2), :] += displacement
            cell_points2[np.isin(cell_indices2, cell_indices1), :] -= displacement
        if self.cell_swapping_probab > 0.0:
            if cell_intensities is None:
                raise ValueError("cell_intensities")
            if cell_clusters is None:
                raise ValueError("cell_clusters")
            cell_swapping_indices1 = self._swap_cells(
                cell_intensities1, cell_clusters1, section_rng, first=True
            )
            cell_swapping_indices2 = self._swap_cells(
                cell_intensities2, cell_clusters2, section_rng, first=False
            )
            cell_intensities1 = cell_intensities1[cell_swapping_indices1]
            cell_intensities2 = cell_intensities2[cell_swapping_indices2]
        if self.cell_division_probab > 0.0:
            cell_points1, cell_division_indices1 = self._divide_cells(
                cell_points1, section_rng, first=True
            )
            cell_points2, cell_division_indices2 = self._divide_cells(
                cell_points2, section_rng, first=False
            )
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[cell_division_indices1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[cell_division_indices2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[cell_division_indices1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[cell_division_indices2]
            cell_indices1 = cell_indices1[cell_division_indices1]
            cell_indices2 = cell_indices2[cell_division_indices2]
        if (
            self.image_scale != (1.0, 1.0)
            or self.image_rotation != 0.0
            or self.image_shear != 0.0
            or self.image_translation != (0.0, 0.0)
        ):
            cell_points1 = self._transform_image(cell_points1, first=True)
            cell_points2 = self._transform_image(cell_points2, first=False)
        if self.shuffle_cells:
            cell_shuffling_indices1 = section_rng.permutation(len(cell_points1))
            cell_shuffling_indices2 = section_rng.permutation(len(cell_points2))
            cell_points1 = cell_points1[cell_shuffling_indices1]
            cell_points2 = cell_points2[cell_shuffling_indices2]
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[cell_shuffling_indices1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[cell_shuffling_indices2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[cell_shuffling_indices1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[cell_shuffling_indices2]
            cell_indices1 = cell_indices1[cell_shuffling_indices1]
            cell_indices2 = cell_indices2[cell_shuffling_indices2]
        return (
            (cell_indices1, cell_points1, cell_intensities1),
            (cell_indices2, cell_points2, cell_intensities2),
        )

    def _occlude_image(
        self,
        cell_points: np.ndarray,
        image_size: tuple[int, int],
        first: Optional[bool] = None,
        axis: int = 0,
    ) -> np.array:
        x0 = 0
        x1 = image_size[axis] - 1
        if first is None or first:
            x1 -= round(image_size[axis] * self.image_occlusion / 2)
        if first is None or not first:
            x0 += round(image_size[axis] * self.image_occlusion / 2)
        return (cell_points[:, axis] < x0) | (cell_points[:, axis] > x1)

    def _transform_image(
        self, cell_points: np.ndarray, first: Optional[bool] = None
    ) -> np.ndarray:
        if first is not None:
            t = AffineTransform(
                scale=(
                    self.image_scale[0] ** (-0.5 if first else 0.5),
                    self.image_scale[1] ** (-0.5 if first else 0.5),
                ),
                rotation=(1 if first else -1) * self.image_rotation / 2,
                shear=(1 if first else -1) * self.image_shear / 2,
                translation=(
                    (1 if first else -1) * self.image_translation[0] / 2,
                    (1 if first else -1) * self.image_translation[1] / 2,
                ),
            )
        else:
            t = AffineTransform(
                scale=self.image_scale,
                rotation=self.image_rotation,
                shear=self.image_shear,
                translation=self.image_translation,
            )
        return t(cell_points)

    def _exclude_cells(
        self,
        num_cells: int,
        section_rng: np.random.Generator,
        first: Optional[bool] = None,
    ) -> np.ndarray:
        d = truncnorm.rvs(
            -self.cell_diameter_mean / self.cell_diameter_std,
            self.cell_diameter_mean / self.cell_diameter_std,
            loc=self.cell_diameter_mean,
            scale=self.cell_diameter_std,
            size=num_cells,
            random_state=section_rng,
        )
        if first is not None:
            d *= 2  # equivalent to dividing section thickness by 2
        s = np.ceil(d / self.section_thickness)
        return section_rng.random(size=num_cells) < 1.0 / s

    def _swap_cells(
        self,
        cell_data: np.ndarray,
        cell_clusters: np.ndarray,
        section_rng: np.random.Generator,
        first: Optional[bool] = None,
    ) -> np.ndarray:
        ind = np.arange(len(cell_data))
        p = self.cell_swapping_probab
        if first is not None:
            p /= 2
        n = section_rng.binomial(len(cell_data), p)
        if n == 0:
            return ind
        for i in section_rng.choice(len(cell_data), size=n, replace=False):
            m = cell_clusters == cell_clusters[i]
            j = section_rng.choice(len(cell_data), p=m / np.sum(m))
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def _divide_cells(
        self,
        cell_points: np.ndarray,
        section_rng: np.random.Generator,
        first: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.cell_division_probab
        if first is not None:
            p /= 2
        n = section_rng.binomial(len(cell_points), p)
        ind = section_rng.integers(len(cell_points), size=n)
        radii = section_rng.uniform(low=0.0, high=np.pi, size=n)
        dists = section_rng.normal(
            loc=self.cell_division_dist_mean,
            scale=self.cell_division_dist_std,
            size=n,
        )
        deltas = dists[:, None] * np.column_stack((np.cos(radii), np.sin(radii)))
        cell_division_indices = np.concatenate(
            (np.delete(np.arange(len(cell_points)), ind), ind, ind)
        )
        cell_points = np.concatenate(
            (
                np.delete(cell_points, ind, axis=0),
                cell_points[ind] + deltas,
                cell_points[ind] - deltas,
            )
        )
        return cell_points, cell_division_indices

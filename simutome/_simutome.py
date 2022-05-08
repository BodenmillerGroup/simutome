import sys
from typing import Generator, Optional, Union

import numpy as np
from scipy.stats import truncnorm
from skimage.transform import AffineTransform


Section = tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]


class Simutome:
    def __init__(
        self,
        image_occlusion: float = 0.0,
        image_scale: tuple[float, float] = (1.0, 1.0),
        image_rotation: float = 0.0,
        image_shear: float = 0.0,
        image_translation: tuple[float, float] = (0.0, 0.0),
        exclude_cells: bool = False,
        exclude_cells_swap: float = 0.0,
        section_thickness: Optional[float] = None,
        cell_diameter_mean: Optional[float] = None,
        cell_diameter_std: Optional[float] = None,
        displace_cells: bool = False,
        cell_displacement_var: Optional[float] = None,
        cell_division_probab: float = 0.0,
        cell_division_dist_mean: Optional[float] = None,
        cell_division_dist_std: Optional[float] = None,
        shuffle_cells: bool = True,
        seed=None,
    ) -> None:
        if image_occlusion < 0.0 or image_occlusion >= 1.0:
            raise ValueError("image_occlusion")
        if image_scale[0] <= 0.0 or image_scale[1] <= 0.0:
            raise ValueError("image_scale")
        if exclude_cells_swap < 0.0 or exclude_cells_swap > 1.0:
            raise ValueError("exclude_cells_swap")
        if exclude_cells and (section_thickness is None or section_thickness <= 0.0):
            raise ValueError("section_thickness")
        if exclude_cells and (cell_diameter_mean is None or cell_diameter_mean <= 0.0):
            raise ValueError("cell_diameter_mean")
        if exclude_cells and (cell_diameter_std is None or cell_diameter_std <= 0.0):
            raise ValueError("cell_diameter_std")
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
        self.image_occlusion = image_occlusion
        self.image_scale = image_scale
        self.image_rotation = image_rotation
        self.image_shear = image_shear
        self.image_translation = image_translation
        self.exclude_cells = exclude_cells
        self.exclude_cells_swap = exclude_cells_swap
        self.section_thickness = section_thickness
        self.cell_diameter_mean = cell_diameter_mean
        self.cell_diameter_std = cell_diameter_std
        self.displace_cells = displace_cells
        self.cell_displacement_var = cell_displacement_var
        self.cell_division_probab = cell_division_probab
        self.cell_division_dist_mean = cell_division_dist_mean
        self.cell_division_dist_std = cell_division_dist_std
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
        cell_points = cell_points.copy()
        cell_intensities = None
        if cell_intensities is not None:
            cell_intensities = cell_intensities.copy()
        cell_clusters = None
        if cell_clusters is not None:
            cell_clusters = cell_clusters.copy()
        cell_indices = np.arange(len(cell_points))
        new_cell_mask = np.zeros(len(cell_points), dtype=bool)
        if self.image_occlusion > 0.0:
            if image_size is None:
                raise ValueError("image_size")
            image_occlusion_mask = self._occlude_image(cell_points, image_size)
            cell_indices = cell_indices[~image_occlusion_mask]
            new_cell_mask = new_cell_mask[~image_occlusion_mask]
            cell_points = cell_points[~image_occlusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~image_occlusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~image_occlusion_mask]
        if self.exclude_cells:
            cell_exclusion_mask, cell_swapping_mask = self._exclude_cells(
                len(cell_points), None, section_rng
            )
            if cell_intensities is not None:
                for i in np.nonzero(cell_swapping_mask)[0]:
                    m = cell_clusters == cell_clusters[i]
                    j = section_rng.choice(len(cell_indices), p=m / np.sum(m))
                    new_cell_mask[i] = new_cell_mask[j] = True
                    cell_intensities[i], cell_intensities[j] = (
                        cell_intensities[j],
                        cell_intensities[i],
                    )
            cell_indices = cell_indices[~cell_exclusion_mask]
            new_cell_mask = new_cell_mask[~cell_exclusion_mask]
            cell_points = cell_points[~cell_exclusion_mask]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[~cell_exclusion_mask]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[~cell_exclusion_mask]
        if self.displace_cells:
            cell_points += section_rng.multivariate_normal(
                np.zeros(2),
                np.eye(2) * self.cell_displacement_var,
                size=len(cell_points)
            )
        if self.cell_division_probab > 0.0:
            divided_cell_order, new_cell_points = self._divide_cells(
                cell_indices, None, cell_points, None, section_rng
            )
            cell_indices = cell_indices[divided_cell_order]
            new_cell_mask = new_cell_mask[divided_cell_order]
            cell_points = new_cell_points
            if cell_intensities is not None:
                cell_intensities = cell_intensities[divided_cell_order]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[divided_cell_order]
        if (
            self.image_scale != (1.0, 1.0)
            or self.image_rotation != 0.0
            or self.image_shear != 0.0
            or self.image_translation != (0.0, 0.0)
        ):
            cell_points = self._transform_image(cell_points)
        if self.shuffle_cells:
            shuffled_cell_order = section_rng.permutation(len(cell_points))
            cell_indices = cell_indices[shuffled_cell_order]
            new_cell_mask = new_cell_mask[shuffled_cell_order]
            cell_points = cell_points[shuffled_cell_order]
            if cell_intensities is not None:
                cell_intensities = cell_intensities[shuffled_cell_order]
            if cell_clusters is not None:
                cell_clusters = cell_clusters[shuffled_cell_order]
        return cell_indices, new_cell_mask, cell_points, cell_intensities

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
        cell_points1 = cell_points.copy()
        cell_points2 = cell_points.copy()
        cell_intensities1 = None
        cell_intensities2 = None
        if cell_intensities is not None:
            cell_intensities1 = cell_intensities.copy()
            cell_intensities2 = cell_intensities.copy()
        cell_clusters1 = None
        cell_clusters2 = None
        if cell_clusters is not None:
            cell_clusters1 = cell_clusters.copy()
            cell_clusters2 = cell_clusters.copy()
        cell_indices1 = np.arange(len(cell_points1))
        cell_indices2 = np.arange(len(cell_points2))
        new_cell_mask1 = np.zeros(len(cell_points1), dtype=bool)
        new_cell_mask2 = np.zeros(len(cell_points2), dtype=bool)
        if self.image_occlusion > 0.0:
            if image_size is None:
                raise ValueError("image_size")
            image_occlusion_mask1 = self._occlude_image(
                cell_points1, image_size, where="left"
            )
            image_occlusion_mask2 = self._occlude_image(
                cell_points2, image_size, where="right"
            )
            cell_indices1 = cell_indices1[~image_occlusion_mask1]
            cell_indices2 = cell_indices2[~image_occlusion_mask2]
            new_cell_mask1 = new_cell_mask1[~image_occlusion_mask1]
            new_cell_mask2 = new_cell_mask2[~image_occlusion_mask2]
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
        if self.exclude_cells:
            (
                cell_exclusion_mask1,
                cell_exclusion_mask2,
                cell_swapping_mask1,
                cell_swapping_mask2,
            ) = self._exclude_cells(cell_indices1, cell_indices2, section_rng)
            if cell_intensities1 is not None:
                for i in np.flatnonzero(cell_swapping_mask1):
                    m = cell_clusters1 == cell_clusters1[i]
                    j = section_rng.choice(len(cell_indices1), p=m / np.sum(m))
                    new_cell_mask1[i] = new_cell_mask1[j] = True
                    cell_intensities1[i], cell_intensities1[j] = (
                        cell_intensities1[j],
                        cell_intensities1[i],
                    )
            if cell_intensities2 is not None:
                for i in np.flatnonzero(cell_swapping_mask2):
                    m = cell_clusters2 == cell_clusters2[i]
                    j = section_rng.choice(len(cell_indices2), p=m / np.sum(m))
                    new_cell_mask2[i] = new_cell_mask2[j] = True
                    cell_intensities2[i], cell_intensities2[j] = (
                        cell_intensities2[j],
                        cell_intensities2[i],
                    )
            cell_indices1 = cell_indices1[~cell_exclusion_mask1]
            cell_indices2 = cell_indices2[~cell_exclusion_mask2]
            new_cell_mask1 = new_cell_mask1[~cell_exclusion_mask1]
            new_cell_mask2 = new_cell_mask2[~cell_exclusion_mask2]
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
        if self.displace_cells:
            shared_cell_indices = np.intersect1d(cell_indices1, cell_indices2)
            delta = section_rng.multivariate_normal(
                np.zeros(2),
                np.eye(2) * self.cell_displacement_var,
                size=len(shared_cell_indices)
            )
            cell_points1[np.isin(cell_indices1, shared_cell_indices), :] += 0.5 * delta
            cell_points2[np.isin(cell_indices2, shared_cell_indices), :] -= 0.5 * delta
        if self.cell_division_probab > 0.0:
            (
                divided_cell_order1,
                divided_cell_order2,
                cell_points1,
                cell_points2,
            ) = self._divide_cells(
                cell_indices1,
                cell_indices2,
                cell_points1,
                cell_points2,
                section_rng,
            )
            cell_indices1 = cell_indices1[divided_cell_order1]
            cell_indices2 = cell_indices2[divided_cell_order2]
            new_cell_mask1 = new_cell_mask1[divided_cell_order1]
            new_cell_mask2 = new_cell_mask2[divided_cell_order2]
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[divided_cell_order1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[divided_cell_order2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[divided_cell_order1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[divided_cell_order2]
        if (
            self.image_scale != (1.0, 1.0)
            or self.image_rotation != 0.0
            or self.image_shear != 0.0
            or self.image_translation != (0.0, 0.0)
        ):
            cell_points1 = self._transform_image(cell_points1, half=True, reverse=True)
            cell_points2 = self._transform_image(cell_points2, half=True, reverse=False)
        if self.shuffle_cells:
            shuffled_cell_order1 = section_rng.permutation(len(cell_points1))
            shuffled_cell_order2 = section_rng.permutation(len(cell_points2))
            cell_indices1 = cell_indices1[shuffled_cell_order1]
            cell_indices2 = cell_indices2[shuffled_cell_order2]
            new_cell_mask1 = new_cell_mask1[shuffled_cell_order1]
            new_cell_mask2 = new_cell_mask2[shuffled_cell_order2]
            cell_points1 = cell_points1[shuffled_cell_order1]
            cell_points2 = cell_points2[shuffled_cell_order2]
            if cell_intensities1 is not None:
                cell_intensities1 = cell_intensities1[shuffled_cell_order1]
            if cell_intensities2 is not None:
                cell_intensities2 = cell_intensities2[shuffled_cell_order2]
            if cell_clusters1 is not None:
                cell_clusters1 = cell_clusters1[shuffled_cell_order1]
            if cell_clusters2 is not None:
                cell_clusters2 = cell_clusters2[shuffled_cell_order2]
        return (
            (cell_indices1, new_cell_mask1, cell_points1, cell_intensities1),
            (cell_indices2, new_cell_mask2, cell_points2, cell_intensities2),
        )

    def _occlude_image(
        self,
        cell_points: np.ndarray,
        image_size: tuple[int, int],
        where: Optional[str] = None,
    ) -> np.array:
        x0 = 0
        x1 = image_size[0] - 1
        if where is None or where == "left":
            x0 += round(image_size[0] * self.image_occlusion / 2)
        if where is None or where == "right":
            x1 -= round(image_size[0] * self.image_occlusion / 2)
        return (cell_points[:, 0] < x0) | (cell_points[:, 0] > x1)

    def _exclude_cells(
        self,
        cell_indices1: np.ndarray,
        cell_indices2: Optional[np.ndarray],
        section_rng: np.random.Generator,
    ) -> Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        cell_indices = cell_indices1
        if cell_indices2 is not None:
            cell_indices = np.union1d(cell_indices1, cell_indices2)
        d = truncnorm.rvs(
            -self.cell_diameter_mean / self.cell_diameter_std,
            self.cell_diameter_mean / self.cell_diameter_std,
            loc=self.cell_diameter_mean,
            scale=self.cell_diameter_std,
            size=len(cell_indices),
            random_state=section_rng,
        )
        s = np.ceil(d / self.section_thickness)
        selected_cell_indices = cell_indices[
            section_rng.random(size=len(cell_indices)) < 1.0 / s
        ]
        n_exclude = int(len(selected_cell_indices) * (1 - self.exclude_cells_swap))
        n_swap = int(len(selected_cell_indices) * self.exclude_cells_swap / 2)
        excluded_cell_indices = selected_cell_indices[:n_exclude]
        swapped_cell_indices = selected_cell_indices[n_exclude : n_exclude + n_swap]
        if cell_indices2 is not None:
            excluded_cell_indices1, excluded_cell_indices2 = self._split_cell_indices(
                excluded_cell_indices,
                cell_indices1,
                cell_indices2,
                section_rng,
            )
            swapped_cell_indices1, swapped_cell_indices2 = self._split_cell_indices(
                swapped_cell_indices,
                cell_indices1,
                cell_indices2,
                section_rng,
            )
            cell_exclusion_mask1 = np.isin(cell_indices1, excluded_cell_indices1)
            cell_exclusion_mask2 = np.isin(cell_indices2, excluded_cell_indices2)
            cell_swapping_mask1 = np.isin(cell_indices1, swapped_cell_indices1)
            cell_swapping_mask2 = np.isin(cell_indices2, swapped_cell_indices2)
            return (
                cell_exclusion_mask1,
                cell_exclusion_mask2,
                cell_swapping_mask1,
                cell_swapping_mask2,
            )
        cell_exclusion_mask = np.isin(cell_indices, excluded_cell_indices)
        cell_swapping_mask = np.isin(cell_indices, swapped_cell_indices)
        return cell_exclusion_mask, cell_swapping_mask

    def _divide_cells(
        self,
        cell_indices1: np.ndarray,
        cell_indices2: Optional[np.ndarray],
        cell_points1: np.ndarray,
        cell_points2: Optional[np.ndarray],
        section_rng: np.random.Generator,
    ) -> Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        cell_indices = cell_indices1
        cell_points = cell_points1
        if cell_indices2 is not None and cell_points2 is not None:
            cell_indices = np.union1d(cell_indices1, cell_indices2)
            cell_points = None
        n = section_rng.binomial(len(cell_indices), p=self.cell_division_probab)
        divided_cell_indices = section_rng.choice(cell_indices, size=n, replace=False)
        radii = section_rng.uniform(low=0.0, high=np.pi, size=n)
        dists = section_rng.normal(
            loc=self.cell_division_dist_mean,
            scale=self.cell_division_dist_std,
            size=n,
        )
        deltas = dists[:, None] * np.column_stack((np.cos(radii), np.sin(radii)))
        if cell_indices2 is not None and cell_points2 is not None:
            divided_cell_indices1, divided_cell_indices2 = self._split_cell_indices(
                divided_cell_indices, cell_indices1, cell_indices2, section_rng
            )
            deltas1 = deltas[: len(divided_cell_indices1), :]
            deltas2 = deltas[len(divided_cell_indices1) :, :]
            m1 = np.isin(cell_indices1, divided_cell_indices1)
            m2 = np.isin(cell_indices2, divided_cell_indices2)
            divided_cell_order1 = np.arange(len(cell_indices1))
            divided_cell_order2 = np.arange(len(cell_indices2))
            divided_cell_order1 = np.concatenate(
                (
                    divided_cell_order1[~m1],
                    divided_cell_order1[m1],
                    divided_cell_order1[m1],
                )
            )
            divided_cell_order2 = np.concatenate(
                (
                    divided_cell_order2[~m2],
                    divided_cell_order2[m2],
                    divided_cell_order2[m2],
                )
            )
            divided_cell_points1 = np.concatenate(
                (
                    cell_points1[~m1, :],
                    cell_points1[m1, :] - deltas1,
                    cell_points1[m1, :] + deltas1,
                )
            )
            divided_cell_points2 = np.concatenate(
                (
                    cell_points2[~m2, :],
                    cell_points2[m2, :] - deltas2,
                    cell_points2[m2, :] + deltas2,
                )
            )
            return (
                divided_cell_order1,
                divided_cell_order2,
                divided_cell_points1,
                divided_cell_points2,
            )
        m = np.isin(cell_indices, divided_cell_indices)
        divided_cell_order = np.arange(len(cell_indices))
        divided_cell_order = np.concatenate(
            (divided_cell_order[~m], divided_cell_order[m], divided_cell_order[m])
        )
        divided_cell_points = np.concatenate(
            (cell_points[~m, :], cell_points[m, :] - deltas, cell_points[m, :] + deltas)
        )
        return divided_cell_order, divided_cell_points

    def _transform_image(
        self, cell_points: np.ndarray, half: bool = False, reverse: bool = False
    ) -> np.ndarray:
        if half:
            t = AffineTransform(
                scale=(
                    self.image_scale[0] ** (-0.5 if reverse else 0.5),
                    self.image_scale[1] ** (-0.5 if reverse else 0.5),
                ),
                rotation=(1 if reverse else -1) * self.image_rotation / 2,
                shear=(1 if reverse else -1) * self.image_shear / 2,
                translation=(
                    (1 if reverse else -1) * self.image_translation[0] / 2,
                    (1 if reverse else -1) * self.image_translation[1] / 2,
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

    def _split_cell_indices(
        self,
        selected_cell_indices: np.ndarray,
        cell_indices1: np.ndarray,
        cell_indices2: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_cell_indices1 = np.intersect1d(cell_indices1, selected_cell_indices)
        selected_cell_indices2 = np.intersect1d(cell_indices2, selected_cell_indices)
        shared_selected_cell_indices = np.intersect1d(
            selected_cell_indices1, selected_cell_indices2
        )
        n = len(shared_selected_cell_indices) // 2
        if n > 0:
            rng.shuffle(shared_selected_cell_indices)
            selected_cell_indices1 = np.setdiff1d(
                selected_cell_indices1, shared_selected_cell_indices[:n]
            )
            selected_cell_indices2 = np.setdiff1d(
                selected_cell_indices2, shared_selected_cell_indices[n:]
            )
        assert len(selected_cell_indices1) + len(selected_cell_indices2) == len(
            selected_cell_indices
        )
        return selected_cell_indices1, selected_cell_indices2

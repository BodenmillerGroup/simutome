import numpy as np

from skimage.transform import AffineTransform
from typing import Callable, Generator, NamedTuple, Optional, Tuple


class Simutome:
    class CellDisplacementParams(NamedTuple):
        displacement_mean_x: float
        displacement_mean_y: float
        displacement_var_x: float
        displacement_var_y: float

    class CellSplitingParams(NamedTuple):
        splitting_probab: float
        splitting_dist_mean: float
        splitting_dist_std: float

    CellDataTransform = Callable[[np.ndarray, np.random.Generator], np.ndarray]

    class ImageTransformParams(NamedTuple):
        scale_x: float = 1.0
        scale_y: float = 1.0
        rotation_rad: float = 0.0
        shear_rad: float = 0.0
        translation_x: float = 0.0
        translation_y: float = 0.0

    def __init__(
        self,
        img_occlusion_frac: Optional[float] = None,
        cell_exclusion_probab: Optional[float] = None,
        cell_displacement_params: Optional[CellDisplacementParams] = None,
        cell_data_permutation_probab: Optional[float] = None,
        cell_splitting_params: Optional[CellSplitingParams] = None,
        cell_data_transform: Optional[CellDataTransform] = None,
        img_transform_params: Optional[ImageTransformParams] = None,
        shuffle: bool = True,
        seed=None,
    ) -> None:
        self.img_occlusion_frac = img_occlusion_frac
        self.cell_exclusion_probab = cell_exclusion_probab
        self.cell_displacement_params = cell_displacement_params
        self.cell_data_permuation_probab = cell_data_permutation_probab
        self.cell_splitting_params = cell_splitting_params
        self.cell_data_transform = cell_data_transform
        self.img_transform_params = img_transform_params
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed=seed)

    def generate_sections(
        self,
        data: np.ndarray,
        coords: np.ndarray,
        orig_width: int,
        orig_height: int,
        n: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        i = 0
        while n is None or i < n:
            yield self.generate_section(data, coords, orig_width, orig_height)
            i += 1

    def generate_section(
        self,
        data: np.ndarray,
        coords: np.ndarray,
        clusters: np.ndarray,
        orig_img_width: int,
        orig_img_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_ind = np.arange(len(data))
        if self.img_occlusion_frac is not None:
            occlusion_mask = self._occlude_image(
                coords, orig_img_width, orig_img_height
            )
            coords = coords[occlusion_mask]
            data = data[occlusion_mask]
            clusters = clusters[occlusion_mask]
            orig_ind = orig_ind[occlusion_mask]
        if self.cell_exclusion_probab is not None:
            exclusion_mask = self._exclude_cells(len(coords))
            coords = coords[exclusion_mask]
            data = data[exclusion_mask]
            clusters = clusters[exclusion_mask]
            orig_ind = orig_ind[exclusion_mask]
        if self.cell_displacement_params is not None:
            coords = self._displace_cells(coords)
        if self.cell_data_permuation_probab is not None:
            data = self._permute_cell_data(data, clusters)
        if self.cell_splitting_params is not None:
            data, coords, clusters, orig_ind = self._split_cells(
                data, coords, clusters, orig_ind
            )
        if self.cell_data_transform is not None:
            data = self.cell_data_transform(data, self._rng)
        if self.img_transform_params is not None:
            coords = self._transform_image(
                coords, orig_img_width, orig_img_height
            )
        if self.shuffle:
            shuffled_ind = self._rng.permutation(len(coords))
            coords = coords[shuffled_ind]
            data = data[shuffled_ind]
            orig_ind = orig_ind[shuffled_ind]
        return data, coords, orig_ind

    def _occlude_image(
        self,
        coords: np.ndarray,
        orig_img_width: int,
        orig_img_height: int,
    ) -> np.array:
        w = int(round(orig_img_width * (1 - self.img_occlusion_frac) ** 0.5))
        h = int(round(orig_img_height * (1 - self.img_occlusion_frac) ** 0.5))
        x0 = self._rng.integers(orig_img_width - w)
        y0 = self._rng.integers(orig_img_height - h)
        return (
            (coords[:, 0] >= x0)
            & (coords[:, 0] < x0 + w)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] < y0 + h)
        )

    def _transform_image(
        self, coords: np.ndarray, orig_img_width: int, orig_img_height: int
    ) -> np.ndarray:
        center_transform = AffineTransform(
            translation=(-orig_img_width / 2.0, -orig_img_height / 2.0)
        )
        transform = AffineTransform(
            scale=(
                self.img_transform_params.scale_x,
                self.img_transform_params.scale_y,
            ),
            rotation=self.img_transform_params.rotation_rad,
            shear=self.img_transform_params.shear_rad,
            translation=(
                self.img_transform_params.translation_x,
                self.img_transform_params.translation_y,
            ),
        )
        return center_transform.inverse(transform(center_transform(coords)))

    def _exclude_cells(self, num_cells: int) -> np.ndarray:
        return self._rng.choice(
            [False, True],
            size=num_cells,
            p=[self.cell_exclusion_probab, 1.0 - self.cell_exclusion_probab],
        )

    def _displace_cells(self, coords: np.ndarray) -> np.ndarray:
        mean = [
            self.cell_displacement_params.displacement_mean_x,
            self.cell_displacement_params.displacement_mean_y,
        ]
        cov = np.eye(2) * [
            self.cell_displacement_params.displacement_var_x,
            self.cell_displacement_params.displacement_var_y,
        ]
        return coords + self._rng.multivariate_normal(
            mean, cov, size=len(coords)
        )

    def _permute_cell_data(
        self, data: np.ndarray, clusters: np.ndarray
    ) -> np.ndarray:
        n = self._rng.binomial(len(data), self.cell_data_permuation_probab)
        if n == 0:
            return data, clusters
        data_copy = data.copy()
        for i in self._rng.choice(len(data), size=n, replace=False):
            candidate_mask = clusters == clusters[i]
            j = self._rng.choice(
                len(data), p=candidate_mask / np.sum(candidate_mask)
            )
            data_copy[i], data_copy[j] = data[j], data[i]
        return data_copy

    def _split_cells(
        self,
        data: np.ndarray,
        coords: np.ndarray,
        clusters: np.ndarray,
        orig_ind: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = self._rng.choice(
            [True, False],
            size=len(coords),
            p=[
                self.cell_splitting_params.splitting_probab,
                1.0 - self.cell_splitting_params.splitting_probab,
            ],
        )
        r = self._rng.uniform(low=0.0, high=np.pi, size=np.sum(m))
        d = self._rng.normal(
            loc=self.cell_splitting_params.splitting_dist_mean,
            scale=self.cell_splitting_params.splitting_dist_std,
            size=np.sum(m),
        )
        delta = d * np.column_stack((np.cos(r), np.sin(r)))
        coords = np.concatenate(coords[~m], coords[m] + delta, coords[m] - delta)
        data = np.concatenate(data[~m], data[m], data[m])
        clusters = np.concatenate(clusters[~m], clusters[m], clusters[m])
        orig_ind = np.concatenate(orig_ind[~m], orig_ind[m], orig_ind[m])
        return data, coords, clusters, orig_ind

"""This module implements a distributional relation of distances from locations to map features."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from itertools import product
from pathlib import Path

# Third Party
from numpy import array, clip, mean, unravel_index, var, vectorize
from scipy.stats import multivariate_normal
from shapely.strtree import STRtree

# ProMis
from promis.geo import CartesianLocation, CartesianMap, LocationType, PolarMap, RasterBand
from promis.models import Gaussian


class Distance:

    """TODO"""

    def __init__(self, mean: RasterBand, variance: RasterBand, location_type: LocationType) -> None:
        # Setup attributes
        self.mean = mean
        self.variance = variance
        self.location_type = location_type

        # TODO: Find better treatment of zero variance
        self.variance.data = clip(self.variance.data, 0.001, None)

    def __lt__(self, value: float) -> RasterBand:
        probabilities = RasterBand(
            self.mean.data.shape, self.mean.origin, self.mean.width, self.mean.height
        )

        for x, y in product(range(self.mean.data.shape[0]), range(self.mean.data.shape[1])):
            probabilities.data[x, y] = Gaussian(self.mean.data[x, y].reshape((1, 1)), self.variance.data[x, y].reshape((1, 1))).cdf(array([value]))

        return probabilities

    def __gt__(self, value: float) -> RasterBand:
        probabilities = self < value
        probabilities.data = 1 - probabilities.data

        return probabilities

    def split(self) -> "list[list[Distance]] | Distance":
        mean_splits = self.mean.split()
        variance_splits = self.variance.split()

        if isinstance(mean_splits, RasterBand):
            return self

        return [
            [
                Distance(mean_splits[0][0], variance_splits[0][0], self.location_type),
                Distance(mean_splits[0][1], variance_splits[0][1], self.location_type),
            ],
            [
                Distance(mean_splits[1][0], variance_splits[1][0], self.location_type),
                Distance(mean_splits[1][1], variance_splits[1][1], self.location_type),
            ],
        ]

    def save_as_plp(self, path: Path) -> None:
        with open(path, "w") as plp_file:
            plp_file.write(self.to_distributional_clauses())

    def to_distributional_clauses(self) -> str:
        code = ""
        for index in product(range(self.mean.data.shape[0]), range(self.mean.data.shape[1])):
            code += self.index_to_distributional_clause(index)

        return code

    def index_to_distributional_clause(self, index: tuple[int, int]) -> str:
        # Build code
        feature_name = self.location_type.name.lower()
        relation = f"distance(row_{index[1]}, column_{index[0]}, {feature_name})"
        distribution = f"normal({self.mean.data[index]}, {self.variance.data[index]})"

        return f"{relation} ~ {distribution}.\n"

    @classmethod
    def from_map(
        cls,
        map_: PolarMap | CartesianMap,
        location_type: LocationType,
        resolution: tuple[int, int],
        number_of_samples: int = 50,
    ) -> "Distance | None":
        # Setup attributes
        cartesian_map = map_ if isinstance(map_, CartesianMap) else map_.to_cartesian()

        # If map is empty return
        if cartesian_map.features is None:
            return None

        # Get all relevant features
        features = [
            feature for feature in cartesian_map.features if feature.location_type == location_type
        ]
        if not features:
            return None

        # Construct an STR tree per collection of varitions of features
        str_trees = [
            STRtree(
                [
                    feature.sample()[0].geometry
                    if feature.distribution is not None
                    else feature.geometry
                    for feature in features
                ]
            )
            for _ in range(number_of_samples)
        ]

        # Initialize raster-bands for mean and variance
        mean = RasterBand(
            resolution, cartesian_map.origin, cartesian_map.width, cartesian_map.height
        )
        variance = RasterBand(
            resolution, cartesian_map.origin, cartesian_map.width, cartesian_map.height
        )

        # Compute parameters of normal distributions for each location
        for i, location in enumerate(mean.cartesian_locations.values()):
            index = unravel_index(i, mean.data.shape)
            mean.data[index], variance.data[index] = cls.extract_parameters(location, str_trees)

        # Create and return Distance object
        return cls(mean, variance, location_type)

    @staticmethod
    def extract_parameters(location: CartesianLocation, str_trees: list[STRtree]) -> float:
        """Computes mean and variance for the distance of a location to all geometries of a type.

        Args:
            location: The location to compute the distance statistic for
            str_trees: Random variations of the features of a map indexible by an STRtree each

        Returns:
            Mean and variance of a normal distribution modeling the distance of this location to the
            nearest map features of specified type
        """

        distances = []
        for str_tree in str_trees:
            distances.append(
                location.geometry.distance(
                    str_tree.geometries.take(str_tree.nearest(location.geometry))
                )
            )

        return mean(distances), var(distances)

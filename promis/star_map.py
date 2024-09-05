"""This module contains a class for handling probabilistic, semantic and geospatial data."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import cache
from itertools import product
from pickle import dump, load
from time import time

# Third Party
from numpy import array, sort, unique
from numpy.random import choice
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from shapely import STRtree
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize

# ProMis
from promis.geo import CartesianCollection, CartesianLocation, CartesianMap, CartesianRasterBand
from promis.logic.spatial import Distance, Over, Relation
from promis.logic.spatial.depth import Depth


class StaRMap:
    """A Statistical Relational Map.

    Args:
        target: The collection of points to output for each relation
        uam: The uncertainty annotated map as generator in Cartesian space
        method: The method to approximate parameters from a set of support points;
            one of {"linear", "nearest", "gaussian_process"}
    """

    def __init__(
        self,
        target: CartesianCollection,
        uam: CartesianMap,
        location_types: list[str | None] | None = None,
        method="linear",
    ) -> "StaRMap":
        """Setup the StaR Map environment representation."""

        # Check parameters
        assert method in [
            "linear",
            "nearest",
            "gaussian_process",
        ], f"StaRMap does not support the method {method}"

        # Setup distance and over relations
        self.target = target
        self.uam = uam
        self.method = method

        if location_types is None:
            self.location_types = list(uam.all_location_types) + [None]
        else:
            self.location_types = location_types

        # Each relation is stored as collection of support points and fitted approximator
        self.relations = {
            "over": defaultdict(
                lambda: {
                    "collection": CartesianCollection(self.target.origin, 2),
                    "approximator": None,
                }
            ),
            "distance": defaultdict(
                lambda: {
                    "collection": CartesianCollection(self.target.origin, 2),
                    "approximator": None,
                }
            ),
            "depth": defaultdict(
                lambda: {
                    "collection": CartesianCollection(self.target.origin, 2),
                    "approximator": None,
                }
            ),
        }

    @staticmethod
    def relation_name_to_class(relation: str) -> type[Over] | type[Distance]:
        match relation:
            case "over":
                return Over
            case "distance":
                return Distance
            case "depth":
                return Depth
            case _:
                raise ValueError(f"Requested unknown relation '{relation}' from StaR Map")

    @staticmethod
    def load(path) -> "StaRMap":
        with open(path, "rb") as file:
            return load(file)

    def save(self, path):
        with open(path, "wb") as file:
            dump(self, file)

    def set_method(self, method: str):
        self.method = method
        self.fit()

    def fit(
        self,
        relations: list[str] | None = None,
        location_types: list[str | None] | None = None,
    ):
        if relations is None:
            relations = self.relations.keys()

        if location_types is None:
            location_types = self.location_types

        # Predict for each value
        for relation, location_type in product(relations, location_types):
            # Not all relations must be present for all location types
            if relation not in self.relations or location_type not in self.relations[relation]:
                continue

            if self.method == "gaussian_process":
                # Setup input scaler
                scaler = StandardScaler().fit(self.target.coordinates())

                # Fit GP to relation data and store approximator
                gaussian_process, _ = self._train_gaussian_process(
                    self.relations[relation][location_type]["collection"], None
                )
                self.relations[relation][location_type]["approximator"] = (gaussian_process, scaler)
            elif self.method == "linear":
                # Get coordinates of overall training samples so far
                coordinates = self.relations[relation][location_type]["collection"].coordinates()

                # Fit linear interpolator for each relation
                self.relations[relation][location_type]["approximator"] = LinearNDInterpolator(
                    coordinates, self.relations[relation][location_type]["collection"].values()
                )
            elif self.method == "nearest":
                # Get coordinates of overall training samples so far
                coordinates = self.relations[relation][location_type]["collection"].coordinates()

                # Fit voronoi interpolator for each relation
                self.relations[relation][location_type]["approximator"] = NearestNDInterpolator(
                    coordinates, self.relations[relation][location_type]["collection"].values()
                )
            else:
                raise f"Unsupported method {self.method} in StaRMap!"

    def get(self, relation: str, location_type: str) -> Distance | Over:
        """Get the computed data for a relation to a location type.

        Args:
            relation: The relation to return, currently 'over' and 'distance' are supported
            location_type: The location type to relate to

        Returns:
            The Collection of computed points for this relation
        """

        parameters = deepcopy(self.target)
        coordinates = parameters.coordinates()

        if self.relations[relation][location_type]["approximator"] is None:
            raise RuntimeError(
                f"Relation {relation} for location type {location_type} has not been fitted yet! "
                "Please call fit() or add_support_points() first, using all relevant location types."
            )

        if self.method == "gaussian_process":
            gp, scaler = self.relations[relation][location_type]["approximator"]
            approximated = gp.predict(scaler.transform(coordinates))
        else:
            approximated = self.relations[relation][location_type]["approximator"](coordinates)

        parameters.data["v0"] = approximated[:, 0]
        parameters.data["v1"] = approximated[:, 1]

        klass = self.relation_name_to_class(relation)
        match relation:
            case "distance" | "over":
                return klass(parameters, location_type)
            case "depth":
                return klass(parameters)

    def all_relations(self) -> list[Relation]:
        relations = []
        for relation_type in self.relations:
            for location_type in self.relations[relation_type]:
                relations.append(self.get(relation_type, location_type))
        return relations

    def _train_gaussian_process(
        self,
        support: CartesianCollection,
        pretrained_gp: tuple[GaussianProcessRegressor, StandardScaler] | None = None,
    ) -> tuple[GaussianProcessRegressor, float]:
        # Fit input scaler on target space
        if pretrained_gp is not None:
            input_scaler = pretrained_gp[1]
        else:
            input_scaler = StandardScaler().fit(self.target.coordinates())

        # Setup kernel and GP
        kernel = 1 * RBF(array([1.0, 1.0])) + WhiteKernel()
        if pretrained_gp is not None:
            kernel.set_params(**(pretrained_gp[0].kernel_.get_params()))

        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, normalize_y=True
        )

        # Fit on support data
        # TODO: This has raised ConvergenceWarnings in the past that where no actual problems
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            start = time()
            gaussian_process.fit(input_scaler.transform(support.coordinates()), support.values())
            elapsed = time() - start

        return gaussian_process, elapsed

    def auto_improve(
        self,
        number_of_random_maps: int,
        number_of_improvement_points: int,
        relations: list[str] | None = None,
        location_types: list[str] | None = None,
    ):
        assert isinstance(
            self.target, CartesianRasterBand
        ), "StaRMap improve currently only works with RasterBand targets!"

        if relations is None:
            relations = self.relations.keys()

        if location_types is None:
            location_types = self.location_types

        for relation, location_type in product(relations, location_types):
            gaussian_process, scaler = self.relations[relation][location_type]["approximator"]

            std = gaussian_process.predict(
                scaler.transform(self.target.coordinates()), return_std=True
            )[1]

            # We decide improvement points for mean and variance separately
            improvement_collection = CartesianCollection(self.target.origin)
            uncertainty = deepcopy(self.target)
            uncertainty.data["v0"] = std[:, 0]
            uncertainty_image = uncertainty.as_image()

            improvement_points = choice(
                uncertainty_image.shape[0] * uncertainty_image.shape[1],
                size=number_of_improvement_points,
                replace=False,
                p=normalize(array([uncertainty_image.ravel()]), norm="l1").ravel(),
            )

            locations = [
                CartesianLocation(
                    uncertainty.data["east"][index],
                    uncertainty.data["north"][index],
                )
                for index in improvement_points
            ]

            improvement_collection.append_with_default(locations, 0.0)
            self.add_support_points(
                improvement_collection, number_of_random_maps, [relation], [location_type]
            )

    def prune(
        self,
        threshold: float,
        relations: list[str] | None = None,
        location_types: list[str] | None = None,
    ):
        if relations is None:
            relations = self.relations.keys()

        if location_types is None:
            location_types = self.location_types

        for relation, location_type in product(relations, location_types):
            coordinates = self.relations[relation][location_type]["collection"].coordinates()
            clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(
                coordinates
            )

            pruning_index = sort(unique(clusters.labels_, return_index=True)[1])
            pruned_coordinates = coordinates[pruning_index]
            pruned_values = self.relations[relation][location_type]["collection"].values()[
                pruning_index
            ]

            self.relations[relation][location_type]["collection"].clear()
            self.relations[relation][location_type]["collection"].append(
                pruned_coordinates, pruned_values
            )

        self.fit(relations, location_types)

    def add_support_points(
        self,
        support: CartesianCollection,
        number_of_random_maps: int,
        relations: list[str] | None = None,
        location_types: list[str | None] | None = None,
    ):
        """Compute distributional clauses.

        Args:
            support: The Collection of points for which the spatial relations will be computed
            number_of_random_maps: How often to sample from map data in order to
                compute statistics of spatial relations
            location_types: Which types of geospatial data to compute
        """

        # TODO: This could be parallelized, as each relation is independent from the others.

        support_coordinates = support.coordinates()
        support_points = array([location.geometry for location in support.to_cartesian_locations()])

        if relations is None:
            relations = self.relations.keys()

        if location_types is None:
            location_types = self.location_types

        @cache
        def sampled_rtrees_for(location_type: str | None) -> list[STRtree]:
            filtered_map: CartesianMap = self.uam.filter(location_type)
            return [
                random_map.to_rtree() for random_map in filtered_map.sample(number_of_random_maps)
            ]

        for relation, location_type in product(relations, location_types):
            # Get all relevant features from map
            if location_type is None:
                continue  # Skip depth, as it is handled separately below
            r_trees = sampled_rtrees_for(location_type)

            # If map had no relevant features, fill with default values
            if r_trees[0].geometries.size == 0:
                self.relations[relation][location_type]["collection"].append_with_default(
                    support_coordinates, (None, None)
                )
            else:
                match relation:
                    case "distance" | "over":
                        # Setup data structures
                        values = self.relation_name_to_class(relation).compute_parameters(
                            support_points, r_trees
                        )

                        # Add to collections
                        self.relations[relation][location_type]["collection"].append(
                            support_coordinates, values
                        )
                    case "depth":
                        pass  # Nothing to do here per location_type, it's handled specially below
                    case _:
                        raise ValueError(f"Requested unknown relation '{relation}' from StaR Map")

        # Depth is a special case, as it is not dependent on the location type
        if any(
            location_type in self.location_types for location_type in Depth.RELEVANT_LOCATION_TYPES
        ):
            self.relations["depth"][None]["collection"].append(
                support_coordinates, Depth.compute_parameters(self.uam, support).T
            )
        else:
            self.relations["depth"][None]["collection"].append_with_default(
                support_coordinates, (None, None)
            )

        self.fit(relations, location_types)

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
from itertools import product
from pickle import dump, load
from re import finditer
from time import time
from typing import Any, TypedDict

# Third Party
from numpy import array, sort, unique, vstack
from numpy.random import choice
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize

# ProMis
from promis.geo import CartesianCollection, CartesianLocation, CartesianMap, RasterBand
from promis.logic.spatial import Depth, Distance, Over, Relation


# TODO make Private?
class RelationInformation(TypedDict):
    collection: CartesianCollection
    approximator: None | object


class StaRMap:
    """A Statistical Relational Map.

    Among others, this holds two types of points: the target points and the support points.
    Initially the value of the relations are determined at the support points.
    To determine the value at the target points, the relations are approximated using the
    support points, e.g., through linear interpolation.
    When solving a ProMis problem, the solution is computed at the target points.

    Note:
        For efficiency reasons, support points can only be given as a regular grid.

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
        method: str = "linear",
    ) -> None:
        """Setup the StaR Map environment representation."""

        # Setup distance and over relations
        self.uam = uam
        self.target = target  # Assumes that self.uam is already set
        self.method = method

        # Each relation is stored as collection of support points and fitted approximator
        self.relations: dict[str, dict[str | None, RelationInformation]]
        self.clear_relations()

    def initialize(self, support: CartesianCollection, number_of_random_maps: int, logic: str):
        """Setup the StaRMap for a given set of support points, number of samples and set of constraints.

        Args:
            support: The support points to be computed
            number_of_random_maps: The number of samples to be used per support point
            logic: The set of constraints deciding which relations are computed
        """

        # Aggregate all relations
        agg = defaultdict(list)
        for relation, location_type in self._get_mentioned_relations(logic):
            agg[relation].append(location_type)

        self.add_support_points(support, number_of_random_maps, agg)

    def clear_relations(self):
        """Clear out the stored relations data."""

        self.relations = {
            "over": defaultdict(self._empty_relation),
            "distance": defaultdict(self._empty_relation),
            "depth": defaultdict(self._empty_relation),
        }

    def _empty_relation(self) -> RelationInformation:
        return {
            # Two values for storing mean and variance
            "collection": CartesianCollection(self.target.origin, 2),
            "approximator": None,
        }

    @staticmethod
    def relation_name_to_class(relation: str) -> Relation:
        if relation not in ["over", "distance", "depth"]:
            raise NotImplementedError(f'Requested unknown relation "{relation}" from StaR Map')

        # TODO: make more elegant/extensible
        match relation:
            case "over":
                return Over
            case "distance":
                return Distance
            case "depth":
                return Depth

    @property
    def relation_types(self) -> set[str]:
        return set(self.relations.keys())

    @property
    def relation_and_location_types(self) -> dict[str, set[str]]:
        return {name: set(info.keys()) for name, info in self.relations.items()}

    @property
    def relation_arities(self) -> dict[str, int]:
        return {name: self.relation_name_to_class(name).arity() for name in self.relation_types}

    @property
    def target(self) -> CartesianCollection:
        return self._target

    @target.setter
    def target(self, target: CartesianCollection) -> None:
        # Validate that target and UAM have the same origin coordinates
        # TODO: Why does PolarLocation not have a __eq__ method?
        if any(target.origin.to_numpy() != self.uam.origin.to_numpy()):
            raise ValueError(
                "StaRMap target and UAM must have the same origin but were: "
                f"{target.origin} and {self.uam.origin}"
            )

        # Actually store the target
        self._target = target

        # Make sure to refit if target changes
        if self.is_fitted:
            self.fit()

    @staticmethod
    def load(path) -> "StaRMap":
        with open(path, "rb") as file:
            return load(file)

    def save(self, path):
        with open(path, "wb") as file:
            dump(self, file)

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        if method not in ["linear", "nearest", "gaussian_process"]:
            raise NotImplementedError(f'StaRMap does not support the method "{method}"')
        self._method = method

        # Make sure to refit if method changes
        if self.is_fitted:
            self.fit()

    def fit(self, what: dict[str, list[str]] | None = None) -> None:
        if what is None:
            what = self.relation_and_location_types  # Inefficient, but works

        # Predict for each value
        for relation, location_types in what.items():
            for location_type in location_types:
                # Not all relations must be present for all location types
                if relation not in self.relations or location_type not in self.relations[relation]:
                    continue  # Nothing to do here

                info = self.relations[relation][location_type]

                match self.method:
                    case "gaussian_process":
                        # Setup input scaler
                        scaler = StandardScaler().fit(self.target.coordinates())

                        # Fit GP to relation data and store approximator
                        gaussian_process, _ = self._train_gaussian_process(info["collection"], None)
                        info["approximator"] = (gaussian_process, scaler)

                    # Could easily be extended to other methods, like spline interpolation
                    case "linear" | "nearest":
                        # If `collection` is a reaster band, this will be particularly efficient
                        info["approximator"] = info["collection"].get_interpolator(self.method)

                    case _:
                        raise NotImplementedError(f"Unsupported method {self.method} in StaRMap!")

    @property
    def is_fitted(self) -> bool:
        # In the beginning, self.relations might not be defined yet
        return hasattr(self, "relations") and all(
            info["approximator"] is not None
            for entries in self.relations.values()
            for info in entries.values()
        )

    def get(self, relation: str, location_type: str) -> Relation:
        """Get the computed data for a relation to a location type.

        Args:
            relation: The relation to return
            location_type: The location type to relate to

        Returns:
            The relation for the given location type
        """

        parameters = deepcopy(self.target)
        coordinates = parameters.coordinates()

        info = self.relations[relation][location_type]
        if self.method == "gaussian_process":
            gp, scaler = info["approximator"]
            approximated = gp.predict(scaler.transform(coordinates))
        else:
            approximated = info["approximator"](coordinates)

        parameters.data["v0"] = approximated[:, 0]
        parameters.data["v1"] = approximated[:, 1]

        return self.relation_name_to_class(relation)(parameters, location_type)

    def get_all(self) -> list[Relation]:
        """Get all the relations for each location type.

        Returns:
            A list of all relations
        """

        relations = self.relations.keys()
        location_types = self.location_types

        return [
            self.get(relation, location_type)
            for relation, location_type in product(relations, location_types)
        ]

    def _get_mentioned_relations(self, logic: str) -> list[tuple[str, str]]:
        """Get all relations mentioned in a logic program.

        Args:
            logic: The logic program

        Returns:
            A list of the (relation_type, location_type) pairs mentioned in the program
        """

        # TODO can it really be None?
        mentioned_relations: list[tuple[str, str | None]] = []

        for name, arity in self.relation_arities.items():
            realtes_to = ",".join([r"\s*((?:'\w*')|(?:\w+))\s*"] * (arity - 1))

            # Prepend comma to first element if not empty
            if realtes_to:
                realtes_to = "," + realtes_to

            for match in finditer(rf"({name})\(X{realtes_to}\)", logic):
                name = match.group(1)
                if name == "landscape":
                    # TODO really?
                    continue  # Ignore landscape relation since it is not part of the StaRMap

                match arity:
                    case 1:
                        mentioned_relations.append((name, None))
                    case 2:
                        location_type = match.group(2)
                        if location_type[0] in "'\"":  # Remove quotes
                            location_type = location_type[1:-1]
                        mentioned_relations.append((name, location_type))
                    case _:
                        raise Exception(f"Only arity 1 and 2 are supported, but got {arity}")

        return mentioned_relations

    def get_from_logic(self, logic: str) -> list[Relation]:
        """Get all relations mentioned in a logic program.

        Args:
            logic: The logic program

        Returns:
            A list of the Relations mentioned in the program
        """
        # TODO make in list comprehension

        relations = []
        for relation_type, location_type in self._get_mentioned_relations(logic):
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
        relations: list[str],
        location_types: list[str],
    ):
        assert isinstance(self.target, RasterBand), (
            "StaRMap auto_improve() currently only works with RasterBand targets!"
        )

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
        relations: list[str],
        location_types: list[str],
    ):
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
        what: dict[str, list[str]] | None = None,
    ):
        """Compute distributional clauses.

        Args:
            support: The Collection of points for which the spatial relations will be computed
            number_of_random_maps: How often to sample from map data in order to
                compute statistics of spatial relations
            what: The spatial relations to compute, as a mapping of relation names to location types
        """

        what = self.relation_and_location_types if what is None else what
        all_location_types = [location_type for types in what.values() for location_type in types]

        for location_type in all_location_types:
            # Get all relevant features from map
            typed_map = self.uam.filter(location_type)

            # Setup data structures
            random_maps = typed_map.sample(number_of_random_maps)
            r_trees = [instance.to_rtree() for instance in random_maps]
            locations = support.to_cartesian_locations()

            for relation, types in what.items():
                if relation not in what or location_type not in types:
                    continue
                else:
                    info = self.relations[relation][location_type]

                # If map had no relevant features, fill with default values
                if not typed_map.features:
                    info["collection"].append_with_default(
                        support.coordinates(),
                        self.relation_name_to_class(relation).empty_map_parameters(),
                    )

                    continue

                try:
                    values = vstack(
                        [
                            self.relation_name_to_class(relation).compute_parameters(
                                location, r_trees
                            )
                            for location in locations
                        ]
                    )

                    # Add to collections
                    info["collection"].append(locations, values)
                except Exception as e:
                    print(
                        f"""StaR Map encountered excpetion {e};
                        {relation} for {location_type} will use default parameters!"""
                    )

                    info["collection"].append_with_default(
                        support.coordinates(),
                        self.relation_name_to_class(relation).empty_map_parameters(),
                    )

        # TODO: see if this is still required
        # # TODO: This could be parallelized, as each relation is independent from the others.

        # Version NEWER ...
        # # TODO: this is a bit of a hack and should be done more elegantly
        # self._support_resolution = support.resolution
        # support_coordinates = support.coordinates()
        # support_points = array([location.geometry for location in support.to_cartesian_locations()])

        # if relations is None:
        #     relations = self.relations.keys()

        # if location_types is None:
        #     location_types = self.location_types

        # @cache
        # def sampled_rtrees_for(location_type: str | None) -> list[STRtree]:
        #     filtered_map: CartesianMap = self.uam.filter(location_type)
        #     return [
        #         random_map.to_rtree() for random_map in filtered_map.sample(number_of_random_maps)
        #     ]
        # ... Version NEWER END

        # support_coordinates = support.coordinates()
        # support_points = array([location.geometry for location in support.to_cartesian_locations()])

        # if relations is None:
        #     relations = self.relations.keys()

        # if location_types is None:
        #     location_types = self.location_types

        # @cache
        # def sampled_rtrees_for(location_type: str | None) -> list[STRtree]:
        #     filtered_map: CartesianMap = self.uam.filter(location_type)
        #     return [
        #         random_map.to_rtree() for random_map in filtered_map.sample(number_of_random_maps)
        #     ]

        # for relation, location_type in product(relations, location_types):
        #     # Get all relevant features from map
        #     if location_type is None:
        #         continue  # Skip depth, as it is handled separately below
        #     r_trees = sampled_rtrees_for(location_type)

        #     # If map had no relevant features, fill with default values
        #     if r_trees[0].geometries.size == 0:
        #         self.relations[relation][location_type]["collection"].append_with_default(
        #             support_coordinates, (None, None)
        #         )
        #     else:
        #         match relation:
        #             case "distance" | "over":
        #                 # Setup data structures
        #                 values = self.relation_name_to_class(relation).compute_parameters(
        #                     support_points, r_trees
        #                 )

        #                 # Add to collections
        #                 self.relations[relation][location_type]["collection"].append(
        #                     support_coordinates, values
        #                 )
        #             case "depth":
        #                 pass  # Nothing to do here per location_type, it's handled specially below
        #             case _:
        #                 raise ValueError(f"Requested unknown relation '{relation}' from StaR Map")

        # # Depth is a special case, as it is not dependent on the location type
        # if any(
        #     location_type in self.location_types for location_type in Depth.RELEVANT_LOCATION_TYPES
        # ):
        #     self.relations["depth"][None]["collection"].append(
        #         support_coordinates, Depth.compute_parameters(self.uam, support).T
        #     )
        # else:
        #     self.relations["depth"][None]["collection"].append_with_default(
        #         support_coordinates, (None, None)
        #     )

        self.fit(what)

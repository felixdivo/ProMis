"""This module implements a distributional relation of distances from locations to map features."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Third Party
from shapely.strtree import STRtree

# ProMis
from promis.geo import CartesianCollection, CartesianLocation
from promis.logic.spatial.relation import ScalarRelation


class Distance(ScalarRelation):
    """The distance relation as Gaussian distribution.

    Args:
        parameters: A collection of points with each having values as [mean, variance]
        location_type: The name of the locations this distance relates to
    """

    def __init__(self, parameters: CartesianCollection, location_type: str) -> None:
        super().__init__(parameters, location_type, "distance")

    @staticmethod
    def compute_relation(location: CartesianLocation, r_tree: STRtree) -> float:
        return location.geometry.distance(r_tree.geometries.take(r_tree.nearest(location.geometry)))

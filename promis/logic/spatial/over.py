"""This module implements a distributional predicate of distances to sets of map features."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Third Party
from numpy import ndarray

# Geometry
from shapely import Point, within
from shapely.strtree import STRtree

# ProMis
from .relation import Relation


class Over(Relation):
    def index_to_distributional_clause(self, index: int) -> str:
        return f"{self.parameters.data['v0'][index]}::over(x_{index}, {self.location_type}).\n"

    @staticmethod
    def compute_relation(locations: ndarray[Point], r_tree: STRtree) -> float:
        return within(locations, r_tree.geometries[r_tree.nearest(locations)])

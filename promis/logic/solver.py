"""The solver module provides functionality to create Probabilistic Mission Landscapes."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library

# Third Party
from problog.program import PrologString
from problog.tasks.dcproblog.parser import DCParser
from problog.tasks.dcproblog.solver import InferenceSolver


class Solver:
    """A solver for HPLP based ProMis.

    Args:
        origin: The origin of the navigation space as PolarLocation
        dimensions: The extend of the navigation area in meters
        resolution: The
    """

    def __init__(self, program: str):
        # Setup DCProblog objects
        self.configuration = {
            "abe_name": "pyro",
            "n_samples": 100,
            "ttype": "float32",
            "device": "cpu",
        }
        self.solver = InferenceSolver(**self.configuration)

        # Setup attributes
        self.program = PrologString(program, parser=DCParser())

    def inference(self) -> list[float]:
        # Setup solver with the program and run inference
        inference = self.solver.probability(self.program, **self.configuration)

        # Unpack queried probabilities
        return [
            # Sometimes, the result is a tensor, so we need to extract the value
            float(probability.value)
            for probability in inference["q"].values()
        ]

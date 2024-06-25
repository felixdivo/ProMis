"""The solver module provides inference over Hybrid Probabilistic Logic Programs."""

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

# ProMis
from promis.geo import PolarLocation, RasterBand
from promis.logic.spatial import Distance, Over, Depth


class Solver:

    """A solver for HPLP based ProMis.

    Args:
        program: The logic program written in ProbLog with distributional clauses
    """

    def __init__(self, program: str):
        # Setup DCProblog objects
        self.configuration = {
            "abe_name": "pyro",
            "n_samples": 50,
            "ttype": "float32",
            "device": "cpu",
        }
        self.solver = InferenceSolver(**self.configuration)

        # Setup attributes
        self.program = PrologString(program, parser=DCParser())

    def inference(self) -> list[float]:
        """Run probabilistic inference on the given program.

        Returns:
            A list of probabilities for all queries in the program
        """

        # Setup solver with the program and run inference
        inference = self.solver.probability(self.program, **self.configuration)

        # Unpack queried probabilities
        results = []
        for _, probability in inference["q"].items():
            results.append(probability.value)

        return results

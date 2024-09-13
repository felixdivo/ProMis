"""The ProMis engine for solving constrained navigation tasks using hybrid probabilistic logic."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from copy import deepcopy
from multiprocessing import Pool

# Third Party
from numpy import array
from rich.progress import track

# ProMis
from promis.geo import CartesianRasterBand
from promis.logic import Solver
from promis.star_map import StaRMap


class ProMis:
    """The ProMis engine to create Probabilistic Mission Landscapes."""

    def __init__(self, star_map: StaRMap) -> None:
        """Setup the ProMis engine.

        Args:
            star_map: The statistical relational map holding the parameters for ProMis
        """

        self.star_map = star_map

    def solve(
        self, logic: str, n_jobs: int = 1, batch_size: int = 1, show_progress: bool = False
    ) -> CartesianRasterBand:
        """Solve the given ProMis problem.

        Args:
            logic: The constraints of the landscape(X) predicate, including its definition
            n_jobs: How many workers to use in parallel
            batch_size: How many pixels to infer at once
            show_progress: Whether to show a progress bar

        Returns:
            The Probabilistic Mission Landscape as well as time to
            generate the code, time to compile and time for inference in seconds.
        """

        # For each point in the target CartesianCollection, we need to run a query
        number_of_queries = len(self.star_map.target.data)
        queries = [f"query(landscape(x_{index})).\n" for index in range(number_of_queries)]

        # This is expensive, so we only do it once
        relations = self.star_map.all_relations()

        # We batch up queries into separate programs
        programs: list[str] = []
        for index in range(0, number_of_queries, batch_size):
            # Define the current batch of indices
            batch = range(index, index + batch_size)

            # Write the background knowledge, queries and parameters to the program
            program = logic + "\n"
            for batch_index in batch:
                if batch_index >= number_of_queries:
                    break

                program += queries[batch_index]

                for relation in relations:
                    program += relation.index_to_distributional_clause(batch_index)

            # Add program to collection
            programs.append(program)

        # Solve in parallel with pool of workers
        with Pool(n_jobs) as pool:
            # Make result of Pool computation into flat list of probabilities
            flattened_data = []

            for batch in track(
                pool.imap(
                    self.run_inference, programs, chunksize=10 if len(programs) > 1000 else 1
                ),
                total=len(programs),
                description="Inference",
                disable=not show_progress,
            ):
                flattened_data.extend(batch)

        # Write results to CartesianCollection and return
        inference_results = deepcopy(self.star_map.target)
        inference_results.data["v0"] = array(flattened_data)
        return inference_results

    @staticmethod
    def run_inference(program):
        return Solver(program).inference()

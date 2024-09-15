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
from re import finditer
from functools import cached_property
from pickle import dump, load

# Third Party
from numpy import array
from rich.progress import track

# ProMis
from promis.geo import CartesianCollection
from promis.logic import Solver
from promis.star_map import StaRMap


class ProMis:
    """The ProMis engine to create Probabilistic Mission Landscapes."""

    def __init__(self, star_map: StaRMap, logic: str) -> None:
        """Setup the ProMis engine.

        Args:
            star_map: The statistical relational map holding the parameters for ProMis
            logic: The constraints of the landscape(X) predicate, including its definition
        """

        self.star_map = star_map
        self.logic = logic

    def solve(
        self,
        n_jobs: int = 1,
        batch_size: int = 1,
        show_progress: bool = False,
        print_first: bool = False,
    ) -> CartesianCollection:
        """Solve the given ProMis problem.

        It searches the provided logic for the used relations and location types and
        only encodes the necessary information for the inference.
        It can further parallelize the inference process over multiple workers and batch
        into fewer solver invocations to speed up computations.

        Args:
            n_jobs: How many workers to use in parallel
            batch_size: How many pixels to infer at once
            show_progress: Whether to show a progress bar
            print_first: Whether to print the first program to stdout

        Returns:
            The Probabilistic Mission Landscape as well as time to
            generate the code, time to compile and time for inference in seconds.
        """

        # For each point in the target CartesianCollection, we need to run a query
        number_of_queries = len(self.star_map.target.data)
        queries = [f"query(landscape(x_{index})).\n" for index in range(number_of_queries)]

        # Determine which relations are mentioned in the logic
        # StaRMap.get() is expensive, so we only do this once
        relations = [
            self.star_map.get(relation_type, location_type)
            for relation_type, location_type in self.mentioned_relations
        ]

        # We batch up queries into separate programs
        programs: list[str] = []
        for index in range(0, number_of_queries, batch_size):
            # Define the current batch of indices
            batch = range(index, index + batch_size)

            # Write the background knowledge, queries and parameters to the program
            program = self.logic + "\n"
            for batch_index in batch:
                if batch_index >= number_of_queries:
                    break

                program += queries[batch_index]

                for relation in relations:
                    program += relation.index_to_distributional_clause(batch_index)

                program += "\n"  # Make it easier to read

            # Add program to collection
            programs.append(program)

            if index == 0 and print_first:
                print(program)

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

    @cached_property
    def mentioned_relations(self) -> list[tuple[str, str | None]]:
        """Determine which relations are mentioned in the logic.

        Args:
            logic: The probabilistic logic to search for relations

        Yields:
            A tuple of all nessesary combinations of the relation and location types as strings
        """

        result: list[tuple[str, str | None]] = []

        for name, arity in self.star_map.relation_arities.items():
            realtes_to = ",".join([r"\s*((?:'\w*')|(?:\w+))\s*"] * (arity - 1))

            # Prepend comma to first element if not empty
            if realtes_to:
                realtes_to = "," + realtes_to

            for match in finditer(rf"({name})\(X{realtes_to}\)", self.logic):
                name = match.group(1)
                if name == "landscape":
                    continue  # Ignore landscape relation since it is not part of the StaRMap

                match arity:
                    case 1:
                        result.append((name, None))
                    case 2:
                        location_type = match.group(2)
                        if location_type[0] in "'\"":  # Remove quotes
                            location_type = location_type[1:-1]
                        result.append((name, location_type))
                    case _:
                        raise Exception(f"Only arity 1 and 2 are supported, but got {arity}")

        return result

    @staticmethod
    def run_inference(program):
        return Solver(program).inference()

    @staticmethod
    def load(path) -> "ProMis":
        with open(path, "rb") as file:
            return load(file)

    def save(self, path):
        with open(path, "wb") as file:
            dump(self, file)

"""Functions to check input types and consistency.
"""
import logging

from numpy.random import RandomState
from networkx import DiGraph, NetworkXError, has_path, shortest_path_length

from params import heuristics

logger = logging.getLogger(__name__)


def check_arguments(
    num_stops: int = None,
    load_capacity: list = None,
    duration: int = None,
    pricing_strategy: str = None,
    mixed_fleet: bool = None,
    fixed_cost: bool = None,
    G: DiGraph = None,
    vehicle_types: int = None,
    num_vehicles: list = None,
    use_all_vehicles: bool = None,
):
    """Checks if arguments are consistent."""

    # If num_stops/load_capacity/duration are not integers
    if num_stops and (not isinstance(num_stops, int) or num_stops <= 0):
        raise TypeError("Maximum number of stops must be positive integer.")
    if load_capacity:
        for value in load_capacity:
            if not isinstance(value, int) or value <= 0:
                raise TypeError("Load capacity must be positive integer.")
    if duration and (not isinstance(duration, int) or duration < 0):
        raise TypeError("Maximum duration must be positive integer.")
    # strategies = ['Exact', 'BestEdges1', 'BestEdges2', 'BestPaths', 'Hyper']
    # xukuan
    strategies = heuristics + ["Exact"]
    if pricing_strategy not in strategies:
        raise ValueError(
            "Pricing strategy %s is not valid. Pick one among %s"
            % (pricing_strategy, strategies)
        )
    if mixed_fleet:
        if load_capacity and num_vehicles and len(load_capacity) != len(num_vehicles):
            raise ValueError(
                "Input arguments load_capacity and num_vehicles must have same dimension."
            )
        if load_capacity and fixed_cost and len(load_capacity) != len(fixed_cost):
            raise ValueError(
                "Input arguments load_capacity and fixed_cost must have same dimension."
            )
        if num_vehicles and fixed_cost and len(num_vehicles) != len(fixed_cost):
            raise ValueError(
                "Input arguments num_vehicles and fixed_cost must have same dimension."
            )
        for (i, j) in G.edges():
            if not isinstance(G.edges[i, j]["cost"], list):
                raise TypeError(
                    "Cost attribute for edge (%s,%s) should be of type list"
                )
            if len(G.edges[i, j]["cost"]) != vehicle_types:
                raise ValueError(
                    "Cost attribute for edge (%s,%s) has dimension %s, should have dimension %s."
                    % (i, j, len(G.edges[i, j]["cost"]), vehicle_types)
                )
    if use_all_vehicles:
        if not num_vehicles:
            logger.warning("num_vehicles = None, use_all_vehicles ignored")


def check_seed(seed):
    """Check whether given seed can be used to seed a numpy.random.RandomState
    :return: numpy.random.RandomState (seeded if seed given)
    """
    if seed is None:
        return RandomState()
    elif isinstance(seed, int):
        return RandomState(seed)
    elif isinstance(seed, RandomState):
        return seed
    else:
        raise TypeError("{} cannot be used to seed".format(seed))

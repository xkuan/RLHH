import logging
from collections import deque, Counter
from time import time
from typing import List, Union

import numpy as np
from networkx import DiGraph, shortest_path, add_path
from subproblem_cspy import _SubProblemCSPY
from hyper_heuristic import _HyperHeuristic
from master_solve_pulp import _MasterSolvePulp
from preprocessing import get_num_stops_upper_bound

from data import VCSPDataSet
from params import action_space
# possible values are: WARNING, INFO, DEBUG, ...
# (see https://docs.python.org/3/library/logging.html#logging-levels)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleCrewScheduling:

    def __init__(
            self,
            G=None,
            num_vehicles=None,
    ):
        self.G = G
        self.num_vehicles=num_vehicles

        # Solving parameters
        self.masterproblem: _MasterSolvePulp = None
        self.hyper_heuristic: _HyperHeuristic = None
        self.routes: List = []
        self.comp_time = None

        # Input solving parameters
        self._solver: str = None
        self._time_limit: int = None
        self._pricing_strategy: str = None
        self._cspy: bool = None
        self._elementary: bool = None
        self._dive: bool = None
        self._greedy: bool = None
        self._max_iter: int = None
        self._run_exact = None  # iterations after which the exact algorithm is run

        # parameters for column generation stopping criteria
        self._start_time = None
        self._more_routes = None
        self._iteration = 0  # current iteration
        self._no_improvement = 0  # iterations after with no change in obj func
        self._lower_bound = []
        # Parameters for initial solution and preprocessing
        self._max_capacity: int = None
        self._vehicle_types: Union[List, int] = 1
        self._initial_routes = []
        self._preassignments = []
        self._dropped_nodes = []
        # Parameters for final solution
        self._best_value = None
        self._best_routes = []
        self._best_routes_as_graphs = []
        # others
        # self.instance_name: str = ''
        self.num_customer: int = 0
        self.print_info: bool = False
        self.log_file: str = None
        self.min_resource: List = []
        self.max_resource: List = []
        self.resources: List = []
        self._duplicate_num: List = []
        self.fixed_driver_cost: int = 0
        self.shifts = None

    def initialize(
        self,
        data: VCSPDataSet,
        pricing_strategy = "BestEdges1",
        cspy = True,
        elementary = False,
        time_limit = None,
        solver = "cbc",
        dive = False,
        greedy = False,
        max_iter = None,
        run_exact = 1,
        # heuristic_only = False,
        print_info=False,
        log_file=None
    ):
        self.G = data.G
        self.shifts = data.shifts
        self.fixed_driver_cost = data.fixed_driver_cost
        self.num_customer = data.num_shift
        self.print_info = print_info
        self.log_file = log_file
        self.min_resource = [0, data.min_working_time, 0, 0]
        self.max_resource = [10, data.max_working_time, data.max_driving_time, data.max_driving_time_without_break]
        self.resources = [
            "shift_num",
            "working_time",
            "driving_time",
            "driving_time_without_break"
        ]

        # set solving attributes
        self._more_routes = True
        self._solver = solver
        self._time_limit = time_limit
        self._pricing_strategy = pricing_strategy
        self._cspy = cspy
        self._elementary = elementary if not dive else True
        self._dive = False
        self._greedy = greedy
        self._max_iter = max_iter
        self._run_exact = run_exact
        # self._heuristic_only = heuristic_only
        if self._pricing_strategy == "Hyper":
            self.hyper_heuristic = _HyperHeuristic()

        self._start_time = time()

        # Pre-processing
        self._pre_solve()

        self._initialize(solver)

        if self.log_file:
            with open (self.log_file, 'a') as file:
                file.write(f"heuristic:  {pricing_strategy}\n")

    def _pre_solve(self):
        self.num_vehicles = []
        self._vehicle_types = 1
        num_stops = get_num_stops_upper_bound(self.G, self.max_resource[2])
        self.num_stops = num_stops

    def _initialize(self, solver):
        """Initialization with feasible solution."""
        self._get_initial_solution()
        # Keep a (deep) copy of the graph
        self._H = self.G.to_directed()
        # self.num_vehicles[0] = max(self.num_vehicles[0], len(self._initial_routes))
        # Initial routes are converted to digraphs
        self._convert_initial_routes_to_digraphs()
        # Init master problem
        self.masterproblem = _MasterSolvePulp(
            self.G,
            self._routes_with_node,
            self._routes,
            # self.drop_penalty,
            self.num_vehicles,
            # self.use_all_vehicles,
            # self.periodic,
            # self.minimize_global_span,
            solver,
        )

    def _add_node_right(self, node, route: List, resource):
        if len(route) == 0:
            i, j = "Source", node
        else:
            i, j = route[-1], node
        new_res = list(resource)    # 直接=的话是浅拷贝
        try:
            edge_data = self.G.edges[i,j]
        except KeyError:
            return new_res, False
        # shift_num
        new_res[0] += edge_data["shift_num"]
        # working_time
        new_res[1] += edge_data["working_time"]
        # driving_time
        new_res[2] += edge_data["driving_time"]
        # driving_time_without_break
        if edge_data["drive_without_break"]:
            new_res[3] += edge_data["driving_time"]
        else:
            new_res[3] = 0
        upper_feasibility = np.array(new_res) <= np.array(self.max_resource)
        feasibility = upper_feasibility.all()
        # lower_feasibility = np.array(new_res) >= np.array(self.min_resource)
        # feasibility = upper_feasibility.all() and lower_feasibility.all()
        return new_res, feasibility

    def _add_node_left(self, node, route: List, resource=None):
        new_route = [node] + route
        new_res, feasibility = self._cal_route_resources(new_route)
        return new_res, feasibility

    def _cal_route_resources(self, route: List):
        if route[0] == "Source":
            route = route[1:]
        else:
            route = route + ["Sink"]
        resource = [0]*len(self.resources)
        for i, node in enumerate(route):
            new_res, feasibility = self._add_node_right(node, route[:i], resource)
            if feasibility:
                resource = new_res
            else:
                return None, False
        return resource, True

    def _get_solution_round_trips(self, k=2):
        # logger.info("get_solution_round_trips")
        routes = []
        cumulative_resources = []
        all_nodes = [v for v in self.G.nodes() if v not in ["Source", "Sink"]]
        residual_num = len(all_nodes) % k
        if residual_num > 0:
            new_res = [0] * len(self.resources)
            routes.append([])
            cumulative_resources.append(new_res)
            for node in all_nodes:
                new_res, feasibility = self._add_node_right(node, routes[0], new_res)
                if feasibility:
                    all_nodes.remove(node)
                    routes[0].append(node)
                    cumulative_resources[0] = new_res
                    if len(routes[0]) == residual_num:
                        break

        all_nodes = np.array(all_nodes).reshape((k, int(len(all_nodes) / k))).T
        for nodes in all_nodes:
            route = ["Source"]
            resource = [0] * len(self.resources)
            for node in nodes:
                new_res, feasibility = self._add_node_right(node, route, resource)
                if not feasibility:
                    return [], [], False
                route.append(node)
                resource = new_res
            # add Sink
            resource, feasibility = self._add_node_right("Sink", route, resource)
            route.append("Sink")
            routes.append(route)
            cumulative_resources.append(resource)

        return routes, cumulative_resources, True

    def _get_solution_heuristic(self):
        routes = [[]]
        cumulative_resources = [[0]*len(self.resources)]
        residual_node = deque([v for v in self.G.nodes() if v not in ["Source", "Sink"]])
        j = 0  # 记录被抽过点的路径
        while len(residual_node) > 0:
            current_node = residual_node.popleft()
            done = False
            # 先尝试将当前点加入到一个已经存在的路径中
            for i in range(len(routes)):
                new_res, feasibility = self._add_node_right(
                    current_node, routes[i], cumulative_resources[i])
                if feasibility:
                    routes[i].append(current_node)
                    cumulative_resources[i] = new_res
                    done = True
                    break
            # 如果失败了，那就添加一个新的路径
            while not done:
                # 尝试将该点作为第一个点
                try:
                    edge_data = self.G.edges["Source", current_node]
                    routes.append([current_node])
                    cumulative_resources.append([
                        1, edge_data["working_time"], edge_data["driving_time"], edge_data["driving_time"]])
                    done = True
                # 如果该点时间太晚了以至于不能作为第一个点，那暂且先将其放回队列
                except KeyError:
                    residual_node.appendleft(current_node)
                    # 尝试将现有路径中的第二个点或中间点拿出来重开一个新路径
                    # node_index = max(1, int(len(routes[j]) / 2))
                    node_index = -2
                    try:
                        current_node = routes[j].pop(node_index)
                        edge_data = self.G.edges["Source", current_node]
                        routes.append([current_node])
                        cumulative_resources.append([
                            1, edge_data["working_time"], edge_data["driving_time"], edge_data["driving_time"]])
                        done = True
                        j += 1
                    # 如果失败, 或者路径本身只有一个点
                    except (KeyError, IndexError):
                        # routes[i] = routes[i][:node_index] + [current_node] + routes[i][node_index:]
                        # residual_node.appendleft(current_node)
                        # 如果所有路径都失败，则返回 False
                        # if i == len(routes) - 1:
                        return [], [], False
                    # except IndexError:   # index error（pop的时候点的数量不够）
                    #     break
        for _ in range(3):
            routes, cumulative_resources = self._adjust_lower_feasibility(routes, cumulative_resources)
        return routes, cumulative_resources, True

    def _adjust_lower_feasibility(self, routes, cumulative_resources):
        j = 0   # 记录被抽过点的路径
        for i, route in enumerate(routes):
            lower_feasibility = np.array(cumulative_resources[i]) >= np.array(self.min_resource)
            if not lower_feasibility.all():
                done = False
                while not done:
                    if j < len(routes) or len(routes[j]) < 2:  # 确保不会拿走第一个点
                        return routes, cumulative_resources
                    current_node = routes[j].pop(-1)
                    new_res, feasibility = self._add_node_right(
                        current_node, routes[i], cumulative_resources[i])
                    if feasibility:
                        routes[i].append(current_node)
                        cumulative_resources[i] = new_res
                        done = True
                    else:   # 还是同一个点，我们尝试将其加到左边
                        new_res, feasibility = self._add_node_left(
                            current_node, routes[i], cumulative_resources[i])
                        if feasibility:
                            routes[i] = [current_node] + routes[i]
                            cumulative_resources[i] = new_res
                            done = True
                        else:
                            routes[j] = routes[j] + [current_node] # + routes[j][node_index:]
                    j += 1

        return routes, cumulative_resources

    def _get_initial_solution(self):
        # routes, cumul_res, feasibility = self._get_solution_heuristic()
        # if not feasibility:
        #     for k in [2, 3, 4]:
        #         routes, cumul_res, feasibility = self._get_solution_round_trips(k)
        #         if feasibility:
        #             break

        # 这一段是为了确保代码不会出错
        edge_data = self.G.edges["Source", 1]
        sign_on_time = edge_data["working_time"] - edge_data["driving_time"]
        for v in self.G.nodes():
            if v in ["Source", "Sink"] or ("Source", v) in self.G.edges():
                continue
            self.G.add_edge('Source', v,
                            type='sign_on',
                            cost=[self.fixed_driver_cost + sign_on_time],
                            shift_num=1,  # 终点是 shift 则为1
                            working_time=self.G.nodes[v]['duration'] + sign_on_time,
                            driving_time=self.G.nodes[v]['duration'],
                            drive_without_break=1,
                            )

        # if not feasibility:  # 兜底
        routes, cumul_res, feasibility = self._get_solution_round_trips(1)

        # for resource in cumul_res:
        #     lower_feasibility = np.array(resource) >= np.array(self.min_resource)
        #     feasibility = feasibility and lower_feasibility.all()
        # if not feasibility:
        #     logger.info("初始解下可行性：%s" % feasibility)

        self._initial_routes = [["Source"] + route + ["Sink"] for route in routes] if "Sink" not in routes[0] else routes
        self._initial_consumed_resources = cumul_res
        # if self.print_info:
        #     print(self._initial_routes)

    def _convert_initial_routes_to_digraphs(self):
        """
        Converts list of initial routes to list of Digraphs.
        By default, initial routes are computed with the first feasible vehicle type.
        """
        self._routes = []
        self._routes_with_node = {}
        for route_id, r in enumerate(self._initial_routes, start=1):
            total_cost = 0
            G = DiGraph(name=route_id)
            edges = list(zip(r[:-1], r[1:]))
            for (i, j) in edges:
                if (i, j) not in self.G.edges():
                    print(f"({i}, {j}) not in self.G.edges()")
                edge_cost = self.G.edges[i, j]["cost"][0]
                G.add_edge(i, j, cost=edge_cost)
                total_cost += edge_cost
            G.graph["cost"] = total_cost
            G.graph["vehicle_type"] = 0
            G.graph["consumed_resources"] = self._initial_consumed_resources[int(route_id-1)]
            self._routes.append(G)
            for v in r[1:-1]:
                if v in self._routes_with_node:
                    self._routes_with_node[v].append(G)
                else:
                    self._routes_with_node[v] = [G]

    @property
    def best_value(self):
        """Returns value of best solution found."""
        return sum(self.best_routes_cost.values())

    @property
    def best_routes(self):
        """
        Returns dict of best routes found.
        Keys : route_id; values : list of ordered nodes from Source to Sink."""
        return self._best_routes

    @property
    def best_routes_cost(self):
        """Returns dict with route ids as keys and route costs as values."""
        cost = {}
        for route in self.best_routes:
            edges = list(zip(self.best_routes[route][:-1], self.best_routes[route][1:]))
            cost[route] = sum(self._H.edges[i, j]["cost"][0] for (i, j) in edges)
        return cost

    @property
    def best_routes_resource(self):
        return self._best_routes_resource

    def solve(self, post_process: bool=False):

        self._column_generation()

        # Solve as MIP
        final_step = not post_process
        _, _ = self.masterproblem.solve(
            relax=False, time_limit=self._get_time_remaining(mip=True), final_step=final_step
        )
        (
            self._best_value,
            self._best_routes_as_graphs,
        ) = self.masterproblem.get_total_cost_and_routes(relax=False)
        if self.log_file:
            with open (self.log_file, 'a') as file:
                file.write(f"INFO: total cost:  {self._best_value}\n")

        if post_process:
            self._post_process()
        else:
            self._best_routes_as_node_lists()

    def _column_generation(self):
        while self._more_routes:
            # Generate good columns
            self._find_columns()
            # Stop if time limit is passed
            if (
                isinstance(self._get_time_remaining(), float)
                and self._get_time_remaining() == 0.0
            ):
                logger.info("time up !")
                break
            # Stop if no improvement limit is passed or max iter exceeded
            if self._no_improvement > 1000 or (
                self._max_iter and self._iteration >= self._max_iter
            ):
                break

    def _find_columns(self):
        """Solves masterproblem and pricing problem."""
        # Solve restricted relaxed master problem
        start = time()
        duals, relaxed_cost = self.masterproblem.solve(
            relax=True, time_limit=self._get_time_remaining()
        )
        self._lower_bound.append(relaxed_cost)
        if self.print_info:
            logger.info("iteration %s, %.2f, %.4f" % (self._iteration, relaxed_cost, time()-self._start_time))
        if self.log_file:
            with open (self.log_file, 'a') as file:
                file.write("iteration %s, %.2f, %.4f\n" % (self._iteration, relaxed_cost, time()-self._start_time))

        pricing_strategy = self._get_next_pricing_strategy()
        # xukuan, 11.20
        self._current_pricing_strategy = pricing_strategy
        self.duals = duals
        RMP_time = time() - start

        # One subproblem per vehicle type

        start = time()
        for vehicle in range(self._vehicle_types):
            # Solve pricing problem with randomised greedy algorithm
            # if self._greedy:
            #     subproblem = self._def_subproblem(duals, vehicle, greedy=True)
            #     self.routes, self._more_routes = subproblem.solve(n_runs=20)
            #     # Add initial_routes
            #     if self._more_routes:
            #         for r in (
            #             r
            #             for r in self.routes
            #             if r.graph["name"] not in self.masterproblem.y
            #         ):
            #             self.masterproblem.update(r)

            # Continue searching for columns
            self._more_routes = False
            self._more_routes, self._use_excat = self._solve_subproblem_with_heuristic(
                pricing_strategy=pricing_strategy, vehicle=vehicle, duals=duals
            )

            if self._more_routes:
                self.routes[-1].graph["heuristic"] = pricing_strategy
                self.masterproblem.update(self.routes[-1])
                break
            elif self._pricing_strategy == "Hyper":
                self.hyper_heuristic.end_time = time()

        # Keep track of convergence rate and update stopping criteria parameters
        self._iteration += 1
        if self._iteration > 1 and self._lower_bound[-2] == self._lower_bound[-1]:
            self._no_improvement += 1
        else:
            self._no_improvement = 0

        SP_time = time() - start
        return RMP_time, SP_time

    def _solve_subproblem_with_heuristic(
        self,
        pricing_strategy=None,
        vehicle=None,
        duals=None,
        route=None,
    ):
        """Solves pricing problem with input heuristic"""
        more_columns = False
        use_exact = False
        if more_columns:
        # if self._pricing_strategy == "Hyper":
            if pricing_strategy == "BestPaths":
                more_columns = self._attempt_solve_best_paths(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestEdges1":
                more_columns = self._attempt_solve_best_edges1(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestEdges2":
                more_columns = self._attempt_solve_best_edges2(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "Exact":
                more_columns = self._attempt_solve_exact(
                    vehicle=vehicle, duals=duals, route=route
                )
        # old approach
        else:
            if pricing_strategy == "BestPaths":
                more_columns = self._attempt_solve_best_paths(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestEdges1":
                more_columns = self._attempt_solve_best_edges1(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestEdges2":
                more_columns = self._attempt_solve_best_edges2(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestEdges3":
                more_columns = self._attempt_solve_best_edges3(
                    vehicle=vehicle, duals=duals, route=route
                )
            elif pricing_strategy == "BestNodes":
                more_columns = self._attempt_solve_best_nodes(
                    vehicle=vehicle, duals=duals, route=route
                )
            if pricing_strategy == "Exact" or not more_columns:     # 这里与上面不一样
                # xukuan, 10.19, 15:35
                # pricing_strategy 保留了最开始尝试的启发式方法
                use_exact = True
                more_columns = self._attempt_solve_exact(
                    vehicle=vehicle, duals=duals, route=route
                )
        return more_columns, use_exact

    def _attempt_solve_best_paths(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for k_shortest_paths in [11, 15, 19]:
            subproblem = self._def_subproblem(
                duals,
                vehicle,
                route,
                "BestPaths",
                k_shortest_paths,
            )
            self.routes, self._more_routes = subproblem.solve(
                self._get_time_remaining()
            )
            more_columns = self._more_routes
            if more_columns:
                break
        else:
            self._more_routes = True
        return more_columns

    def _attempt_solve_best_edges1(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for alpha in [0.5, 0.9]:
            subproblem = self._def_subproblem(
                duals,
                vehicle,
                route,
                "BestEdges1",
                alpha,
            )
            self.routes, self._more_routes = subproblem.solve(
                self._get_time_remaining()
            )
            more_columns = self._more_routes
            if more_columns:
                break
        else:
            self._more_routes = True
        return more_columns

    def _attempt_solve_best_edges2(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for ratio in [0.1]:
            subproblem = self._def_subproblem(
                duals,
                vehicle,
                route,
                "BestEdges2",
                ratio,
            )
            self.routes, self._more_routes = subproblem.solve(
                self._get_time_remaining(),
            )
            more_columns = self._more_routes
            if more_columns:
                break
        else:
            self._more_routes = True

        return more_columns

    def _attempt_solve_best_edges3(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for ratio in [0.5, 0.7, 0.9]:
            subproblem = self._def_subproblem(
                duals,
                vehicle,
                route,
                "BestEdges3",
                ratio,
            )
            self.routes, self._more_routes = subproblem.solve(
                self._get_time_remaining(),
            )
            more_columns = self._more_routes
            if more_columns:
                break
        else:
            self._more_routes = True

        return more_columns

    def _attempt_solve_best_nodes(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for alpha in [0.1, 0.3, 0.5]:
            subproblem = self._def_subproblem(
                duals,
                vehicle,
                route,
                "BestNodes",
                alpha,
            )
            self.routes, self._more_routes = subproblem.solve(
                self._get_time_remaining(),
            )
            more_columns = self._more_routes
            if more_columns:
                break
        else:
            self._more_routes = True
        return more_columns

    def _attempt_solve_exact(self, vehicle=None, duals=None, route=None):
        subproblem = self._def_subproblem(duals, vehicle, route)
        self.routes, self._more_routes = subproblem.solve(self._get_time_remaining())
        return self._more_routes

    def _get_next_pricing_strategy(self):
        """Return the appropriate pricing strategy based on input parameters"""
        # pricing_strategy = None
        if (
            self._pricing_strategy == "Hyper"
            and self._no_improvement != self._run_exact
        ):
            self._no_improvement = self._iteration
            relaxed_cost = self._lower_bound[-1]
            if self._iteration == 0:
                pricing_strategy = "BestPaths"
                self.hyper_heuristic.init(relaxed_cost)
            else:
                # Get the active paths and the frequency list per heuristic
                self._update_hyper_heuristic(relaxed_cost)
                pricing_strategy = self.hyper_heuristic.pick_heuristic()
        elif self._no_improvement == self._run_exact:
            # self._no_improvement = 0
            pricing_strategy = "Exact"
        else:
            pricing_strategy = self._pricing_strategy
        return pricing_strategy

    def _update_hyper_heuristic(self, relaxed_cost: float):
        best_paths, best_paths_freq = self.masterproblem.get_heuristic_distribution()
        self.hyper_heuristic.current_performance(
            new_objective_value=relaxed_cost,
            produced_column=self._more_routes,
            active_columns=best_paths_freq,
        )
        self.hyper_heuristic.move_acceptance()
        self.hyper_heuristic.update_parameters(self._iteration, self._no_improvement)

    def _get_time_remaining(self, mip: bool = False):
        """
        Modified to avoid over time in subproblems.

        Returns:
            - None if no time limit set.
            - time remaining (in seconds) if time remaining > 0 and mip = False
            - 5 if time remaining < 5 and mip = True
            - 0 if time remaining < 0
        """
        if self._time_limit:
            remaining_time = self._time_limit - (time() - self._start_time)
            if mip:
                return max(10, remaining_time)
            if remaining_time > 0:
                return remaining_time
            return 0.0
        return None

    def _def_subproblem(
        self,
        duals,
        vehicle_type,
        route=None,
        pricing_strategy="Exact",
        pricing_parameter=None,
        # greedy=False,
    ):
        """Instanciates the subproblem."""

        # if self._cspy:
        # With cspy
        subproblem = _SubProblemCSPY(
            self.G,
            duals,
            self._routes_with_node,
            self._routes,
            vehicle_type,
            route,
            self.num_stops,
            # self.load_capacity,
            # self.duration,
            # self.time_windows,
            # self.pickup_delivery,
            # self.distribution_collection,
            pricing_strategy,
            pricing_parameter,
            min_resource=self.min_resource,
            max_resource=self.max_resource,
        )

        return subproblem

    def _best_routes_as_node_lists(self):
        """Converts route as DiGraph to route as node list."""
        self._best_routes = {}
        self._best_routes_list = []
        self._best_routes_resource = {}
        route_id = 1
        for route in self._best_routes_as_graphs:
            node_list = shortest_path(route, "Source", "Sink")
            self._best_routes[route_id] = node_list
            self._best_routes_list.append(node_list[1:-1])
            self._best_routes_resource[route_id] = [int(res) for res in route.graph["consumed_resources"]]
            route_id += 1

    def _post_process(self):
        # 之前是按照每个点的访问次数 >= 1 来解的（不然很可能最优解就是初始解）
        exist_duplicate = True
        num_call_drop_duplicate_nodes = 0
        while exist_duplicate:
            exist_duplicate = self._drop_duplicate_nodes()
            num_call_drop_duplicate_nodes += 1
            if not exist_duplicate and num_call_drop_duplicate_nodes > 1:
                # 说明已经求到了没有重复的最优解,此时重新求一下松弛问题即可(增加了变量)
                _, relaxed_cost = self.masterproblem.solve(
                    relax=True, time_limit=self._get_time_remaining(), final_step=True
                )
                self._lower_bound.append(relaxed_cost)
            if len(self._duplicate_num) > 1 and self._duplicate_num[-1] == self._duplicate_num[-2]:
                logger.info("后处理尚未成功！强行设置访问次数=1")
                # 改不动了，分别求解最后的松弛问题和整数规划问题
                _, relaxed_cost = self.masterproblem.solve(
                    relax=True, time_limit=self._get_time_remaining(), final_step=True
                )
                self._lower_bound.append(relaxed_cost)
                _, relaxed_cost = self.masterproblem.solve(
                    relax=False, time_limit=self._get_time_remaining(mip=True), final_step=True
                )
                (
                    self._best_value,
                    self._best_routes_as_graphs,
                ) = self.masterproblem.get_total_cost_and_routes(relax=False)
                # Convert best routes into lists of nodes
                self._best_routes_as_node_lists()
                return

    def _drop_duplicate_nodes(self):
        """后处理：对于涉及重复访问点的路径，删除重复点生成新的路径"""
        # Convert best routes into lists of nodes
        self._best_routes_as_node_lists()
        node_count = dict(Counter([i for nodes in self._best_routes_list for i in nodes]))
        self._duplicate_num.append(sum(node_count.values()) - len(node_count))
        duplicate_nodes = [node for node in node_count if node_count[node] > 1]

        if len(duplicate_nodes) == 0:
            return False
        for route in self._best_routes_list:
            new_routes = self.create_new_reduced_routes(duplicate_nodes, route)
            for new_route in new_routes:
                new_route, feasibility = self.create_new_route(new_route)
                if feasibility and not any(
                        list(new_route.edges()) == list(r.edges()) for r in self.routes
                ):
                    self.routes.append(new_route)
                    self.routes[-1].graph["heuristic"] = "drop_duplicate"
                    self.masterproblem.update(self.routes[-1])

        _, relaxed_cost = self.masterproblem.solve(
            relax=False, time_limit=self._get_time_remaining(mip=True)
        )
        (
            self._best_value,
            self._best_routes_as_graphs,
        ) = self.masterproblem.get_total_cost_and_routes(relax=False)
        return True

    @staticmethod
    def create_new_reduced_routes(duplicate_nodes: List, route: List):
        remove_list = [[]]
        for node in route:
            if node in duplicate_nodes:
                # if len(remove_list) == 0:
                #     remove_list.append([node])
                # else:
                tmp_list = remove_list.copy()
                for remove_nodes in tmp_list:
                    remove_list.append(remove_nodes + [node])
        new_routes = []
        remove_list.pop(0)
        for remove_nodes in remove_list:
            new_route = route.copy()
            for node in remove_nodes:
                new_route.remove(node)
            new_routes.append(new_route)
        return new_routes

    def create_new_route(self, path):
        """Create new route as DiGraph and add to pool of columns"""
        if path[0] != "Source":
            path = ["Source"] + path + ["Sink"]
        consumed_resources, feasibility = self._cal_route_resources(path)
        if not feasibility:
            return None, False
        route_id = "{}_add".format(len(self.routes) + 1)
        new_route = DiGraph(name=route_id, path=path)
        add_path(new_route, path)
        total_cost = 0
        for (i, j) in new_route.edges():
            edge_cost = self.G.edges[i, j]["cost"][0]
            total_cost += edge_cost
            new_route.edges[i, j]["cost"] = edge_cost
            if i != "Source":
                self._routes_with_node[i].append(new_route)
        new_route.graph["cost"] = total_cost
        new_route.graph["vehicle_type"] = 0
        new_route.graph["consumed_resources"] = consumed_resources
        return new_route, feasibility

    @property
    def iteration(self):
        return self._iteration

    @property
    def num_driver(self):
        return len(self.best_routes)

    @property
    def objval(self):
        return self.best_value - self.num_driver * self.fixed_driver_cost

    def print_solution(self):
        for route_key in self.best_routes:
            route = self.best_routes[route_key]
            resource = self.best_routes_resource[route_key]
            print(f"Driver {route_key}:")
            print(f"  working time: {resource[1]}")
            for shift_id1, shift_id2 in zip(route[:-1], route[1:]):
                if shift_id1 != "Source":
                    shift1 = self.shifts[shift_id1-1]
                    print(f"    shift {shift_id1}: {shift1[1]}-{shift1[2]}, {shift1[5]}")
                if not self.G.edges[shift_id1, shift_id2]["drive_without_break"]:
                    print('    **break**')

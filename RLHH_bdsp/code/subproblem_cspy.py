import logging
from math import floor

from numpy import zeros
from networkx import DiGraph, add_path
from cspy import BiDirectional, REFCallback

from subproblem import _SubProblemBase

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class _MyREFCallback(REFCallback):
    """
    Custom REFs for resources, according to (Desrochers, 1989).
    """

    def __init__(
        self,
        max_res,
        T,
        resources,
    ):
        REFCallback.__init__(self)
        # Set attributes for use in REF functions
        self._max_res = max_res
        self._T = T
        self._resources = resources
        # Set later
        self._sub_G = None
        self._source_id = None
        self._sink_id = None

    def REF_fwd(self, cumul_res, tail, head, edge_res, partial_path, cumul_cost):
        # cumulative 累积的
        new_res = list(cumul_res)
        i, j = tail, head
        edge_data = self._sub_G.edges[i, j]
        # stops / monotone resource
        # new_res[0] += 1
        new_res[0] += edge_data["shift_num"]
        # working_time
        new_res[1] += edge_data["working_time"]
        # driving_time
        new_res[2] += edge_data["driving_time"]
        # driving_time_without_break
        if edge_data["drive_without_break"]:
            new_res[3] += edge_data["driving_time"]
        else:
            new_res[3] = edge_data["driving_time"]

        return new_res


class _SubProblemCSPY(_SubProblemBase):
    def __init__(self, *args, min_resource, max_resource):
        super(_SubProblemCSPY, self).__init__(*args)
        # Resource names
        self.resources = [
            "shift_num",
            "working_time",
            "driving_time",
            "driving_time_without_break"
        ]
        # Set number of resources as attribute of graph
        self.sub_G.graph["n_res"] = len(self.resources)
        # Resource lower and upper bounds
        self.min_res = min_resource
        self.max_res = max_resource
        # Initialize cspy edge attributes
        for edge in self.sub_G.edges(data=True):
            edge[2]["res_cost"] = zeros(len(self.resources))
        # Initialize max feasible arrival time
        self.T = 0
        self.total_cost = None
        # Average length of a path
        self._avg_path_len = 1
        # Iteration counter
        self._iters = 1

    def solve(self, time_limit):
        if not self.run_subsolve:
            return self.routes, False

        time_limit = min(600, time_limit)   # 限制每次求解子问题最多10min
        self.formulate()

        logger.debug("resources = {}".format(self.resources))
        logger.debug("min res = {}".format(self.min_res))
        logger.debug("max res = {}".format(self.max_res))

        more_routes = False

        my_callback = _MyREFCallback(
                self.max_res,
                self.T,
                self.resources,
            )
        direction = "forward"
        elementary = False
        thr = None
        logger.debug(
            f"Solving subproblem using elementary={elementary}, threshold={thr}, direction={direction}"
        )
        alg = BiDirectional(
            self.sub_G,
            self.max_res,
            self.min_res,
            threshold=thr,
            direction=direction,
            time_limit=time_limit,
            elementary=elementary,
            REF_callback=my_callback,
            # pickup_delivery_pairs=self.pickup_delivery_pairs,
        )

        # Pass processed graph
        if my_callback is not None:
            my_callback._sub_G = alg.G
            my_callback._source_id = alg._source_id
            my_callback._sink_id = alg._sink_id
        alg.run()
        logger.debug("subproblem")
        logger.debug("cost = %s", alg.total_cost)
        logger.debug("resources = %s", alg.consumed_resources)

        if alg.total_cost is not None and alg.total_cost < -1e-3:
            new_route = self.create_new_route(alg.path)
            new_route.graph["consumed_resources"] = alg.consumed_resources
            logger.debug(alg.path)
            path_len = len(alg.path)
            if not any(
                    list(new_route.edges()) == list(r.edges()) for r in self.routes
            ):
                more_routes = True
                self.routes.append(new_route)
                self.total_cost = new_route.graph["cost"]
                logger.debug("reduced cost = %s", alg.total_cost)
                logger.debug("real cost = %s", self.total_cost)
                if path_len > 2:
                    self._avg_path_len += (path_len - self._avg_path_len) / self._iters
                    self._iters += 1
            # else:
            #     logger.info("Route already found, finding elementary one")
        return self.routes, more_routes

    def formulate(self):
        """Updates max_res depending on which contraints are active."""
        # Problem specific constraints
        # if self.num_stops:
        #     self.add_max_stops()
        # else:
        #     self.add_monotone()
        for (i, j) in self.sub_G.edges():
            edge_data = self.sub_G.edges[i, j]
            self.sub_G.edges[i, j]["res_cost"][0] = edge_data["shift_num"]
            self.sub_G.edges[i, j]["res_cost"][1] = edge_data["working_time"]
            self.sub_G.edges[i, j]["res_cost"][2] = edge_data["driving_time"]
            self.sub_G.edges[i, j]["res_cost"][3] = edge_data["drive_without_break"]
        # Maximum feasible arrival time
        self.T = max(
            self.sub_G.nodes[v]["upper"]
            + self.sub_G.edges[v, "Sink"]["working_time"]
            for v in self.sub_G.predecessors("Sink")
        )

    def create_new_route(self, path):
        """Create new route as DiGraph and add to pool of columns"""
        e = "elem" if len(set(path)) == len(path) else "non-elem"
        # route_id = "{}_{}".format(len(self.routes) + 1, e)
        route_id = "{}".format(len(self.routes) + 1)
        new_route = DiGraph(name=route_id, path=path)
        add_path(new_route, path)
        total_cost = 0
        for (i, j) in new_route.edges():
            edge_cost = self.sub_G.edges[i, j]["cost"][self.vehicle_type]
            total_cost += edge_cost
            new_route.edges[i, j]["cost"] = edge_cost
            if i != "Source":
                self.routes_with_node[i].append(new_route)
        new_route.graph["cost"] = total_cost
        new_route.graph["vehicle_type"] = self.vehicle_type
        return new_route

    def add_max_stops(self):
        """Updates maximum number of stops."""
        # Change label
        self.resources[0] = "stops"
        # The Sink does not count (hence + 1)
        self.max_res[0] = self.num_stops + 1
        for (i, j) in self.sub_G.edges():
            self.sub_G.edges[i, j]["res_cost"][0] = 1

    def add_monotone(self):
        """Updates monotone resource."""
        # Change label
        self.resources[0] = "mono"
        for (i, j) in self.sub_G.edges():
            self.sub_G.edges[i, j]["res_cost"][0] = 1


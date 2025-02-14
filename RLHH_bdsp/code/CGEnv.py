import logging
import pickle
import random
import time
from collections import namedtuple, deque
from typing import List, Union

import networkx as nx
import numpy as np
import pulp
import torch
from data import VCSPDataSet
from vcsp import VehicleCrewScheduling
from hyper_heuristic import _HyperHeuristic
from master_solve_pulp import _MasterSolvePulp
from entropy_estimators import continuous
from MIP_solver import bus_driver_scheduling
from params import *
# https://github.com/paulbrodersen/entropy_estimators

# from torch_geometric.data import Data as pygData

# possible values are: WARNING, INFO, DEBUG, ...
# (see https://docs.python.org/3/library/logging.html#logging-levels)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CGEnv(VehicleCrewScheduling):

    def __init__(self, device, args, action_space=None):
        super(CGEnv, self).__init__()
        self.device = device
        self.args = args
        self.action_space = action_space
        self.time_limit = args.instance_time_limit
        self.do_post_process = args.post_process
        # self.sub_time_limit = args.sub_time_limit
        # self.n_min = args.n_min
        # self.n_max = args.n_max
        self._action = 1    # BestEdges2
        # self.instance_name: str = ''
        self.num_customer: int = 0

    def _init(self):    # 每次重新读取案例的时候更新一下
        self.SP_times = []
        self.RMP_times = []

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
        self._run_exact = None  # iterations after which the exact algorithm is ran

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
        # self.instance_scale: str = ''
        self.num_customer: int = 0
        self.resources: List = []
        self._duplicate_num: List = []
        self.fixed_driver_cost: int = 0

    def reset(self, run_mode='test', data_path='../data/shift_50_01.csv'):
        self.run_mode = run_mode
        self._init()
        # ============ Problem define ============
        if run_mode == 'train':
            # instance_No = random.randint(1, 30)
            self.num_customer = random.randint(self.args.n_min, self.args.n_max)
            data = VCSPDataSet(n_vertices=self.num_customer)
        else:   # test or debug
            # data_path = '../data/shift_50_01.csv'
            data = VCSPDataSet(path=data_path)
            self.num_customer = data.num_shift
            self.log_file = '../result/detail/RLHH_log_' + data_path.split('/shift_')[-1][:-4] + '.txt'

        self.shifts = data.shifts
        self.initialize(
            data,
            time_limit=self.time_limit,
            pricing_strategy='BestEdges1',
            print_info=self.args.print_info if "print_info" in dir(self.args) else None,
        )

        # ============ First step (get first state) ============
        # self._find_columns()
        state = self._step_return_state()

        return state

    def _step_return_state(self):
        RMP_time, SP_time = self._find_columns()
        state = self._get_state()
        self.RMP_times.append(RMP_time)
        self.SP_times.append(SP_time)
        return state

    def step(self, _action: int):
        self._action = _action
        # self._find_columns()
        state = self._step_return_state()

        done = False
        gap = 1
        if not self._more_routes:
            # print("=========== No more columns found, terminate! ==========")
            done = True
        elif (
            isinstance(self._get_time_remaining(), float)
            and self._get_time_remaining() == 0.0
        ):
            print("=========== Time out, terminate! ==========")
            done = True

        if done:
            self.post_process()
            # gap = (self.best_value - self._lower_bound[-1]) / self.best_value
            gap = self._cal_gap_to_mip_solver()
            # assert gap > -1e-6
            if gap <= -1e-6:
                print("!!!!!!!!!! gap=", gap, "num", self.num_customer)
                # self.print_solution()
            reward = 100 * self.args.alpha ** min(0, -gap) - 1   # 最后一步奖励：[0,99]
        elif self._use_excat:
            reward = -1     # 启发式失败了
        # elif self._no_improvement > 0:
        #     reward = 0      # 找到了新路径但是没有提升解
        # else:
        #     reward = 1      # 找到了新路径并改善解
        else:
            reward = 0      # 找到了新路径

        info = {
            "objval": self._lower_bound[-1],
            "gap": gap,
            "pricing_strategy": self._current_pricing_strategy,
        }

        return state, reward, done, info

    def post_process(self, post_process: bool=False):
        # Solve as MIP
        final_step = not post_process
        _, _ = self.masterproblem.solve(
            # relax=False, time_limit=self._get_time_remaining(mip=True)
            relax=False, time_limit=self._get_time_remaining(mip=True), final_step=final_step
        )
        (
            self._best_value,
            self._best_routes_as_graphs,
        ) = self.masterproblem.get_total_cost_and_routes(relax=False)
        if post_process:
            self._post_process()
        else:
            self._best_routes_as_node_lists()

    def _get_state(self):
        # RLMP 相关特征
        duals = np.array(list(self.duals.values()))
        sols = np.array([pulp.value(self.masterproblem.y[r.graph["name"]]) for r in self.routes[:-1]])
        isnot_int = [abs(sol) > 1e-6 and abs(sol-1) > 1e-6 for sol in sols]
        state = [
            self._lower_bound[-1] / self._lower_bound[0],       # 当前最优值 / 初始目标值
            duals.std() / (duals.mean() - duals.min()),         # 变异系数 Coefficient of Variation
            sum(sols[isnot_int]) / sum(isnot_int) if any(isnot_int) else 0,  # 小数解之和 / 小数解个数
            sum(isnot_int) / len(sols),             # 小数解个数 / 变量个数
        ]
        # 图相关特征
        weight = np.array([d["weight"] for u,v,d in self.G.edges(data=True)
                           if u != "Source" and v != "Sink"])
        cost = np.array([d["cost"][0] for u,v,d in self.G.edges(data=True)
                           if u != "Source" and v != "Sink"])   # working_time
        driving_time = np.array([d["driving_time"] for u,v,d in self.G.edges(data=True)
                           if u != "Source" and v != "Sink"])
        break_time = cost - driving_time
        upper = np.array([d["upper"] for v, d in self.G.nodes(data=True)
                          if v not in ["Source", "Sink"]])
        lower = np.array([d["lower"] for v, d in self.G.nodes(data=True)
                          if v not in ["Source", "Sink"]])
        source_upper = 1440
        tw_span = upper - lower
        # print(continuous.get_h_mvn(weight),continuous.get_h_mvn(cost))
        state.extend([
            weight.std() / (weight.mean() - weight.min()),
            continuous.get_h_mvn(weight) / 10,

            cost.std() / cost.mean(),
            continuous.get_h_mvn(cost) / 10,

            break_time.std() / break_time.mean(),
            continuous.get_h_mvn(break_time) / 10,

            upper.std() / upper.mean(),
            lower.std() / lower.mean(),
            tw_span.std() / tw_span.mean(),
            # tw_span.max()/ source_upper,
            tw_span.mean()/ source_upper,
            # tw_span.min()/ source_upper,
        ])
        # print([round(i,4) for i in state])

        # return torch.tensor(state, device=self.device, dtype=torch.float32)
        return self._get_norm_state(state)

    def _get_norm_state(self, state):
        state = [(state[i] - state_mean[i]) / state_std[i] for i in range(len(state))]
        return torch.tensor(state, device=self.device, dtype=torch.float32)

    def _get_next_pricing_strategy(self):
        if self._iteration == 0:
            return "BestEdges2"
        if self._no_improvement == self._run_exact:
            # self._no_improvement = 0
            return "Exact"
        else:
            return self.action_space[self._action]

    def _cal_gap_to_mip_solver(self):
        # mip_obj = bus_driver_scheduling(shifts=self.shifts, max_num_drivers=self.num_driver,
        #                                 print_soution=False, time_limit=600)
        # if mip_obj <= 1e-6:
        #     return 0
        if self.run_mode == "test":
            mip_obj = 3751 + 12 * self.fixed_driver_cost    # D3QN_train-15.28-11.30
        else:
            add_driver = (self.num_customer - 50) / 3
            mip_obj = 4000 + (10 + add_driver) * self.fixed_driver_cost
            # mip_obj = self._lower_bound[-1]     # - self.num_driver * self.fixed_driver_cost

        gap = (self.best_value - mip_obj) / self.best_value
        return gap

    def _attempt_solve_best_edges2(self, vehicle=None, duals=None, route=None):
        more_columns = False
        for ratio in [0.1, 0.2, 0.3]:
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

    @property
    def n_action(self):
        return len(self.action_space)

    @property
    def new_route(self):
        return self.routes[-1]


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
# 一个命名元组，表示我们环境中的单个转换。它本质上将（状态，动作）对映射到它们的（下一个状态，奖励）结果

class ReplayMemory(object):
    # 一个有界大小的循环缓冲区，用于保存最近观察到的转换。
    def __init__(self, capacity, load_dir=None):
        if load_dir is not None:
            self.memory = pickle.load(open(load_dir, 'rb'))
        else:
            self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, save_dir):
        pickle.dump(self.memory, open(save_dir, 'wb'))

    def __len__(self):
        return len(self.memory)

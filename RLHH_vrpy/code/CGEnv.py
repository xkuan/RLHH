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
from data import SolomonDataSet
from vrp import VehicleRoutingProblem
from hyper_heuristic import _HyperHeuristic
from master_solve_pulp import _MasterSolvePulp
from schedule import _Schedule
from entropy_estimators import continuous
# https://github.com/paulbrodersen/entropy_estimators

from params import probNum
from torch_geometric.data import Data as pygData

# possible values are: WARNING, INFO, DEBUG, ...
# (see https://docs.python.org/3/library/logging.html#logging-levels)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CGEnv(VehicleRoutingProblem):

    def __init__(self, device, args, action_space=None):
        super(CGEnv, self).__init__()
        self.device = device
        self.args = args
        self.action_space = action_space
        self.time_limit = args.instance_time_limit
        # self.sub_time_limit = args.sub_time_limit
        # self.n_min = args.n_min
        # self.n_max = args.n_max
        self._action = 0
        self.instance_name: str = ''
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
        self._schedule: _Schedule = None

        # others
        self.instance_name: str = ''
        self.num_customer: int = 0

    def reset(self, _instance_type="C1", instance=None, run_mode='test'):
        self._init()
        # ============ Problem define ============
        if instance is not None:
            self.instance_name, self.num_customer = instance
        elif run_mode == 'train':
            instance_No = random.randint(1, probNum[_instance_type.lower()])
            self.instance_name = '{}{:02d}'.format(_instance_type.lower(), instance_No)
            self.num_customer = random.randint(self.args.n_min, self.args.n_max)
        else:   # 测试一个固定案例
            self.instance_name, self.num_customer = 'r201', 30
            # self.instance_name, self.num_customer = 'rc107', 25

        self.instance_name = self.instance_name.split('.txt')[0]
        if self.num_customer <= 100:
            data_path = f'../data/{self.instance_name}.txt'
        else:
            type_name = self.instance_name[:-2]
            number = int(self.instance_name[-2:])
            instance_name = f'{type_name.upper()}_{int(self.num_customer/100)}_{number}'
            data_path = f'../data/large/{instance_name}.txt'
        data = SolomonDataSet(path=data_path, n_vertices=self.num_customer)
        self.initialize(
            data,
            time_limit=self.time_limit,
            pricing_strategy='BestEdges1',
            print_info=self.args.print_info,
        )

        # ============ First step (get first state) ============
        # self._find_columns()
        state = self._step_return_state()

        return state

    def _step_return_state(self):
        RMP_time, SP_time = self._find_columns()
        if self.args.net_type in ["MLP", "QNetTwinDuel"]:
            state = self._get_state()
        else:
            state = self._get_state_gnn()
        self.RMP_times.append(RMP_time)
        self.SP_times.append(SP_time)
        return state

    def step(self, _action: int):
        self._action = _action
        # self._find_columns()
        state = self._step_return_state()
        relaxed_objval = self._lower_bound[-1]

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

        # TODO: 把运行时间考虑进奖励中, 考虑最后一步的奖励
        if done:
            self.post_process()
            gap = (self.best_value - relaxed_objval) / self.best_value
            assert gap > -1e-6
            reward = 100 * self.args.alpha ** (-gap) - 1   # 最后一步奖励：最大99
        elif self._use_excat:
            reward = -1     # 启发式失败了
        elif self._no_improvement > 0:
            reward = 0      # 找到了新路径但是没有提升解
        else:
            reward = 1      # 找到了新路径并改善解
        # reward = 1 if self._no_improvement == 0 and not self._use_excat else -1

        info = {
            "objval": relaxed_objval,
            "gap": gap,
            "pricing_strategy": self._current_pricing_strategy,
        }

        return state, reward, done, info

    def post_process(self):
        _, _ = self.masterproblem.solve(
            relax=False, time_limit=self._get_time_remaining(mip=True)
        )
        (
            self._best_value,
            self._best_routes_as_graphs,
        ) = self.masterproblem.get_total_cost_and_routes(relax=False)
        self._post_process(self._solver)

    # TODO: 获取状态
    def _get_state_gnn(self):
        # duals = np.array(list(self.duals.values()))
        node_features, edge_features, edge_index = [], [], []
        source_upper = self.G.nodes(data=True)["Source"]["upper"]
        max_cost = max(d["cost"][0] for u,v,d in self.G.edges(data=True))
        for v, d in self.G.nodes(data=True):
            if v not in ["Source", "Sink"]:
                node_features.append([
                    d["demand"] / self.load_capacity[0],
                    d["lower"] / source_upper,
                    d["upper"] / source_upper,
                    d["tw_span"] / source_upper,
                    self.duals[v] / max_cost
                ])
                # node_features.append(list(np.random.random(5)))
        for u, v, d in self.G.edges(data=True):
            if u != "Source" and v != "Sink":
                edge_features.append([
                    d["cost"][0],
                    # d["time"],    # 跟cost一样的
                    d["weight"],
                    d["pos_weight"],
                ])
                # edge_features.append(list(np.random.random(3)))
                edge_index.append([int(u)-1, int(v)-1])
        state_graph = pygData(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index).T
        )
        state_graph = state_graph.pin_memory()
        state_graph = state_graph.to(self.device, non_blocking=True)
        return state_graph

    def _get_state(self):
        # compute the entropy from the determinant of the multivariate normal distribution:
        # analytic = continuous.get_h_mvn(X)

        # compute the entropy using the k-nearest neighbour approach
        # developed by Kozachenko and Leonenko (1987):
        # kozachenko = continuous.get_h(X, k=3)

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
        # edge_features = np.array([[d["weight"], d["cost"][0]] for i,j,d in self.G.edges(data=True)])
        # weight = edge_features[:, 0]
        # cost = edge_features[:, 1]      # 这样并没有更快
        weight = np.array([d["weight"] for u,v,d in self.G.edges(data=True)
                           if u != "Source" and v != "Sink"])
        cost = np.array([d["cost"][0] for u,v,d in self.G.edges(data=True)
                           if u != "Source" and v != "Sink"])
        upper = np.array([d["upper"] for v, d in self.G.nodes(data=True)
                          if v not in ["Source", "Sink"]])
        lower = np.array([d["lower"] for v, d in self.G.nodes(data=True)
                          if v not in ["Source", "Sink"]])
        source_upper = self.G.nodes(data=True)["Source"]["upper"]
        tw_span = upper - lower
        # print(continuous.get_h_mvn(weight),continuous.get_h_mvn(cost))
        state.extend([
            weight.std() / (weight.mean() - weight.min()),
            continuous.get_h_mvn(weight) / 5,

            cost.std() / cost.mean(),
            continuous.get_h_mvn(cost) / 5,

            upper.std() / upper.mean(),
            lower.std() / lower.mean(),
            tw_span.std() / tw_span.mean(),
            tw_span.max()/ source_upper,
            tw_span.mean()/ source_upper,
            tw_span.min()/ source_upper,
        ])
        # print([round(i,4) for i in state])

        return torch.tensor(state, device=self.device, dtype=torch.float32)

    def _get_next_pricing_strategy(self):
        # TODO: 第一步的action如何选择
        if self._iteration == 0:
            return "BestPaths"
        if self._no_improvement == self._run_exact:
            # self._no_improvement = 0
            return "Exact"
        else:
            return self.action_space[self._action]

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

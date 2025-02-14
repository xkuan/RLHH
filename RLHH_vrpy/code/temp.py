import time
import pandas as pd
import os

from data import SolomonDataSet
from vrp import VehicleRoutingProblem
from params import *

time_limit = 600
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'r205.txt')
# data_path = f'../data/r205.txt'
data = SolomonDataSet(path=data_path, n_vertices=33)

for heuristic in action_space + ["Hyper"] * 3: #
    print("heuristic: ", heuristic)
    VRPTW_model = VehicleRoutingProblem()
    VRPTW_model.initialize(
        data,
        time_limit=time_limit,
        pricing_strategy=heuristic,
        print_info=False,
    )
    start = time.time()
    VRPTW_model.solve(solver="cbc", dive=False)
    tmp = [
        heuristic,
        VRPTW_model.iteration,
        VRPTW_model.best_value,
        time.time() - start,
    ]
    print(tmp)

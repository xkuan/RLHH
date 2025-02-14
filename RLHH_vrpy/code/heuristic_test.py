# r1, r2, n 随机
import random

import numpy as np
import pandas as pd
import time

from data import SolomonDataSet
from vrp import VehicleRoutingProblem
from params import *
from logger import save_log_file
save_log_file(__file__.split('\\')[-1].split('.')[0])


instance_num = 50
def generate_test_instances(num):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

    num_instances_each_type = round(num / 3)
    instance_types = ['c2'] * num_instances_each_type
    instance_types.extend(['r2'] * num_instances_each_type)
    instance_types.extend(['rc2'] * (num - 2 * num_instances_each_type))

    instance_Nos = [random.randint(1, probNum[instance_type]) for instance_type in instance_types]
    instance_names = ['{}{:02d}'.format(instance_types[i], instance_Nos[i]) for i in range(num)]

    customerNums = np.random.randint(25, 35, num)
    return list(zip(instance_names, customerNums))

# df = pd.DataFrame(instances)
# df.columns = ["instance", "n"]
# df.to_csv('../data/test_instances.csv', index=False)

# instances = generate_test_instances(instance_num)
instances = [['c207', 31], ['c205', 31], ['rc207', 30], ['r203', 29], ['r205', 33]]
result = pd.DataFrame(columns=['type', 'instance', 'n', 'method', 'iters', 'objval', 'time'])
for k, (instance_name, customerNum) in enumerate(instances):
    for heuristic in action_space:
        print("\n\n ========== {}/{} prob: {}, n = {}, heuristic: {} ==========".format(
            k+1, instance_num, instance_name, customerNum, heuristic))
        data_path = f'../data/{instance_name}.txt'
        data = SolomonDataSet(path=data_path, n_vertices=customerNum)
        VRPTW_model = VehicleRoutingProblem()
        VRPTW_model.initialize(
            data,
            time_limit=600,
            pricing_strategy=heuristic,
            print_info=False
        )
        start = time.time()
        VRPTW_model.solve(solver="cbc", dive=False)

        result.loc[len(result)] = [
            instance_name[:-2],
            instance_name,
            customerNum,
            heuristic,
            VRPTW_model.iteration,
            VRPTW_model.best_value,
            time.time() - start,
        ]
    result.to_csv("../result/baseline_add.csv", index=False)

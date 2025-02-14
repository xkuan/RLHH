import time
import pandas as pd

from data import SolomonDataSet
from vrp import VehicleRoutingProblem
from params import *

Nums = [200]
time_limit = 3600
result_dir = f"../result/baseline/(c)base_{Nums[0]}-{Nums[-1]}({time_limit}s).csv"
try:
    result = pd.read_csv(result_dir)
except FileNotFoundError:
    result = pd.DataFrame(columns=['No.', 'type', 'instance', 'n', 'method', 'iters', 'objval', 'time'])
k = 0
for customerNum in Nums:
    for prob_type in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:  #
        # for prob in range(probNum[prob_type]):
        for prob in range(5):
            if prob_type == 'c1' or (prob_type == 'c2' and prob < 4):
                continue
            instance_name = '{}{:02d}'.format(prob_type, prob + 1)
            if customerNum <= 100:
                data_path = f'../data/{instance_name}.txt'
            else:
                file_name = f'{prob_type.upper()}_{int(customerNum/100)}_{prob+1}'
                data_path = f'../data/large/{file_name}.txt'
            log_path = f'../result/detail/log_{instance_name}_{customerNum}.txt'
            data = SolomonDataSet(path=data_path, n_vertices=customerNum)
            print("\n\n ========== {}/{} prob: {}, n = {} ==========".format(
                k + 1, 30-9, instance_name, customerNum))
            for heuristic in action_space:  # + ["Hyper"]:
                # if instance_name == "c101" and heuristic in ["BestEdges1", "BestEdges2"]:
                #     continue
                print("heuristic: ", heuristic)
                VRPTW_model = VehicleRoutingProblem()
                VRPTW_model.initialize(
                    data,
                    time_limit=time_limit,
                    pricing_strategy=heuristic,
                    print_info=False,
                    log_file=log_path
                )
                start = time.time()
                VRPTW_model.solve(solver="cbc", dive=False)
                tmp = [
                    k + 1,
                    prob_type,
                    instance_name,
                    customerNum,
                    heuristic,
                    VRPTW_model.iteration,
                    VRPTW_model.best_value,
                    time.time() - start,
                ]
                result.loc[len(result)] = tmp
                result.to_csv(result_dir, index=False)
                print(tmp)
            k += 1



import time
import pandas as pd

from data import VCSPDataSet
from vcsp import VehicleCrewScheduling
from params import *


Nums = [150]
time_limit = 3600

result_dir = f"../result/baseline/(10)base_{Nums[0]}-{Nums[-1]}({time_limit}s).csv"
try:
    result = pd.read_csv(result_dir)
except FileNotFoundError:
    result = pd.DataFrame(columns=['No.', 'instance', 'n', 'method', 'iters', 'driver', 'objval', 'time'])

k = int(len(result) / 6)
for customerNum in Nums:
    for prob in range(10):
        instance_name = '{}_{:02d}'.format(customerNum, prob + 1)
        data_path = f'../data/shift_{instance_name}.csv'
        log_path = f'../result/detail/log_{instance_name}.txt'
        data = VCSPDataSet(path=data_path)
        print("\n\n ========== {}/{} prob: {}, n = {} ==========".format(
                k + 1, 10*len(Nums), instance_name, customerNum))
        for heuristic in action_space + ["Hyper"]: #
            print("heuristic: ", heuristic)
            VCSP_model = VehicleCrewScheduling()
            VCSP_model.initialize(
                data,
                time_limit=time_limit,
                pricing_strategy=heuristic,
                print_info=False,
                log_file=log_path
            )
            start = time.time()
            VCSP_model.solve(post_process=False)
            tmp = [
                k + 1,
                instance_name,
                customerNum,
                heuristic,
                VCSP_model.iteration,
                VCSP_model.num_driver,
                VCSP_model.best_value,
                time.time() - start,
            ]
            print(tmp)
            result.loc[len(result)] = tmp
        k += 1
        result.to_csv(result_dir, index=False)



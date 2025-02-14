import argparse
import os
import random
import time
from itertools import count

import numpy as np
import pandas as pd

import torch
from CGEnv import CGEnv
from params import *
from models import MLP, PDN, GNN
from net import QNetTwinDuel
import result_analysis
from logger import save_log_file

save_log_file(__file__.split('\\')[-1].split('.')[0])

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def get_config(args=None):
    parser = argparse.ArgumentParser(description="RLCG for VRPTW")
    parser.add_argument('--instance_time_limit', type=int, default=3600, help="")
    parser.add_argument('--net_type', type=str, default="MLP", choices=["MLP", "GNN"])
    parser.add_argument('--save_dir', type=str, default="../D3QN_norm")
    parser.add_argument('--instance_size', type=str, default="small", choices=["small", "large"])
    parser.add_argument('--n_min', type=int, default=150, help="min customers of test instance")
    parser.add_argument('--n_max', type=int, default=200, help="max customers of test instance")

    parser.add_argument('--alpha', type=int, default=100, help="")
    parser.add_argument('--epsilon', type=float, default=0.9, help="")
    parser.add_argument('--print_info', type=str_to_bool, default=False, help="")
    parser.add_argument('--post_process', type=str_to_bool, default=False, help="")

    _config = parser.parse_args(args)
    return _config

n_actions = len(action_space)

def predict(model_path, save_dir=None):
    def select_action(_state):
        sample = random.random()
        if sample > config.epsilon:
            policy_net.eval()
            with torch.no_grad():
                if config.net_type == "MLP":
                    out = policy_net(_state.unsqueeze(0)).squeeze()
                else:
                    out = policy_net(_state).squeeze()
                return out.max(0)[1].view(1)
        else:
            return torch.tensor([np.random.choice([1, 3])], device=device, dtype=torch.long)  # BestEdges2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path)
    if config.net_type == "MLP":
        # policy_net = MLP(xdim=num_featrues, ydim=n_actions).to(device)
        policy_net = QNetTwinDuel(mid_dim=64, state_dim=num_featrues, action_dim=n_actions).to(device)
    elif config.net_type == "PDN":
        policy_net = PDN(in_channels=num_node_featrues, out_channels=n_actions, edge_dim=num_edge_featrues).to(device)
    elif config.net_type == "GNN":
        policy_net = GNN(in_channels=num_node_featrues, out_channels=n_actions).to(device)
    else:
        policy_net = None
    policy_net.load_state_dict(checkpoint['net'])
    policy_net.eval()

    env = CGEnv(device, config, action_space)
    result = pd.DataFrame(columns=['No.', 'instance', 'n', 'method', 'iters', 'driver', 'objval', 'time'])

    Nums = [num for num in [50, 75, 100, 150, 200] if config.n_min <= num <= config.n_max]

    k = 0
    for customerNum in Nums:
        for prob in range(10):
            instance_name = '{}_{:02d}'.format(customerNum, prob + 1)
            data_path = f'../data/shift_{instance_name}.csv'
            log_file = f'../result/detail/RLHH_log_{instance_name}.txt'
            print("\n\n ========== {}/{} prob: {}, n = {} ==========".format(
                k + 1, 10 * len(Nums), instance_name, customerNum))
            state = env.reset(data_path=data_path)
            start = time.time()

            for t in count():
                action = select_action(state)
                state, reward, done, info = env.step(action.item())
                # print(action)
                # print('t:{}, reward:{:.4f}, done:{}, objval:{:.4f}, heuristic:{}'.format(
                #     t + 1, reward, done, info['objval'], info['pricing_strategy']))
                with open(log_file, 'a') as file:
                    file.write('t:{}, reward:{:.2f}, done:{}, objval:{:.2f}, heuristic:{}, time:{}\n'.format(
                        t + 1, reward, done, info['objval'], info['pricing_strategy'], time.time() - start))
                if done:
                    tmp = [
                        k+1,
                        instance_name,
                        customerNum,
                        "RLHH",
                        t + 1,
                        env.num_driver,
                        env.best_value,
                        time.time() - start
                    ]
                    print(tmp)
                    result.loc[len(result)] = tmp
                    break

            k += 1
            result.to_csv(save_dir, index=False)


config = get_config()
print(config.__str__())
test_case = config.save_dir
os.makedirs(os.path.join(test_case, 'models'), exist_ok=True)

model_names = [file for file in os.listdir(test_case)
               if 'model' in file
               and not os.path.isdir(os.path.join(test_case,file))
]
model_names = sorted(model_names,key=lambda x: os.path.getmtime(os.path.join(test_case, x)))

for model_name in model_names:
# if len(model_names) > 0:
    # model_name = model_names[-1]    # 时间最近的一个
    print("\n\n=========", model_name)
    # timestamp = model_name.split("_")[-1][:11]
    filename = f"(10){config.n_min}-{config.n_max}_{config.epsilon}_{model_name[6:-4]}"
    save_path = f"../result/{filename}.csv"
    predict(
        # instances=test_instances,
        model_path=f"{test_case}/{model_name}",
        save_dir=save_path
    )
    count_list = result_analysis.main(RLHH_file=filename)
    print("**********", count_list, "**********")

    # result_summary_dir = "../result/result_summary.csv"
    # try:
    #     result_summary = pd.read_csv(result_summary_dir)
    # except FileNotFoundError:
    #     result_summary = pd.DataFrame(
    #         columns=['model', 'bf_50', 'bf_75', 'bf_100', 'bf_sum', 'b_50', 'b_75', 'b_100', 'b_sum'])
    #
    # result_summary.loc[len(result_summary)] = [filename] + count_list
    # result_summary.to_csv(result_summary_dir, index=False)


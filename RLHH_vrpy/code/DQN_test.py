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
    parser.add_argument('--net_type', type=str, default="QNetTwinDuel", choices=["MLP", "GNN", "QNetTwinDuel"])
    parser.add_argument('--save_dir', type=str, default="../D3QN")
    parser.add_argument('--instance_size', type=str, default="large",
                        choices=["small", "large_c", "large_r", "large_rc", "large", "verylarge"])
    # parser.add_argument('--n_min', type=int, default=50, help="min customers of test instance")
    # parser.add_argument('--n_max', type=int, default=50, help="max customers of test instance")

    parser.add_argument('--alpha', type=int, default=100, help="")
    parser.add_argument('--epsilon', type=float, default=0, help="")
    parser.add_argument('--print_info', type=str_to_bool, default=False, help="")
    parser.add_argument('--post_process', type=str_to_bool, default=False, help="")

    _config = parser.parse_args(args)
    return _config

n_actions = len(action_space)

def predict(instances, model_path, save_dir=None):
    def select_action(_state):
        sample = random.random()
        if sample > config.epsilon:
            policy_net.eval()
            with torch.no_grad():
                if config.net_type in ["MLP", "QNetTwinDuel"]:
                    out = policy_net(_state.unsqueeze(0)).squeeze()
                else:
                    out = policy_net(_state).squeeze()
                return out.max(0)[1].view(1)
        else:
            return torch.tensor([1], device=device, dtype=torch.long)  # BestEdges2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path)
    if config.net_type == "MLP":
        policy_net = MLP(xdim=num_featrues, ydim=n_actions).to(device)
        # policy_net = QNetTwinDuel(mid_dim=64, state_dim=num_featrues, action_dim=n_actions).to(device)
    elif config.net_type == "PDN":
        policy_net = PDN(in_channels=num_node_featrues, out_channels=n_actions, edge_dim=num_edge_featrues).to(device)
    elif config.net_type == "GNN":
        policy_net = GNN(in_channels=num_node_featrues, out_channels=n_actions).to(device)
    elif config.net_type == "QNetTwinDuel":
        policy_net = QNetTwinDuel(mid_dim=64, state_dim=num_featrues, action_dim=n_actions).to(device)
    else:
        policy_net = None
    policy_net.load_state_dict(checkpoint['net'])
    policy_net.eval()

    env = CGEnv(device, config, action_space)
    try:
        result = pd.read_csv(save_dir)
    except FileNotFoundError:
        result = pd.DataFrame(columns=['No.', 'type', 'instance', 'n', 'method', 'iters', 'objval', 'gap', 'time'])

    for k, test_instance in enumerate(instances):
        print(test_instance)
        instance_name, customerNum = test_instance
        env.time_limit = 600 if customerNum < 50 else 3600
        log_file = f'../result/detail/RLHH_log_{instance_name}_{customerNum}.txt'
        start = time.time()
        state = env.reset(instance=test_instance)
        for t in count():
            action = select_action(state)
            state, reward, done, info = env.step(action.item())
            # print('t:{}, reward:{:.4f}, done:{}, objval:{:.4f}, heuristic:{}, time:{}'.format(
            #     t + 1, reward, done, info['objval'], info['pricing_strategy'], time.time()-start))
            with open(log_file, 'a') as file:
                file.write('t:{}, reward:{:.2f}, done:{}, objval:{:.2f}, heuristic:{}, time:{}\n'.format(
                    t + 1, reward, done, info['objval'], info['pricing_strategy'], time.time() - start))
            if done:
                tmp = [
                    k+1,
                    instance_name[:-2],
                    instance_name,
                    customerNum,
                    "RLHH",
                    t+1,
                    env.best_value,
                    info['gap'],
                    time.time() - start
                ]
                print(tmp)
                result.loc[len(result)] = tmp
                break

        result.to_csv(save_dir, index=False)


config = get_config()
print(config.__str__())
test_instances = pd.read_csv(f"../result/test_instances_{config.instance_size}.csv").to_numpy()
# test_instances = [
#     ["r101", 25], ["r101", 50], ["r101", 75], ["r101", 100],
#     ["r201", 25], ["r201", 50], ["r201", 75], ["r201", 100],
#     ["c101", 25], ["c101", 50], ["c101", 75], ["c101", 100],
#     ["c201", 25], ["c201", 50], ["c201", 75], ["c201", 100],
#     ["rc101", 25], ["rc101", 50], ["rc101", 75], ["rc101", 100],
#     ["rc201", 25], ["rc201", 50], ["rc201", 75], ["rc201", 100],
# ]
# test_instances = [
#     ["c101", 200], ["c102", 200], ["c103", 200],
#     ["c201", 200], ["c202", 200], ["c203", 200],
#     ["c101", 400], ["c102", 400], ["c103", 400],
#     ["c201", 400], ["c202", 400], ["c203", 400],
#     ["c101", 600], ["c102", 600], ["c103", 600],
#     ["c201", 600], ["c202", 600], ["c203", 600],
# ]

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
    filename = f"{config.instance_size}_{model_name[6:-4]}"
    save_path = f"../result/{filename}.csv"
    predict(
        instances=test_instances,
        model_path=f"{test_case}/{model_name}",
        save_dir=save_path
    )

    # count_list = result_analysis.main(RLHH_file=filename, baseline_filename="large_instances_to_r106_100")
    # print("**********", count_list, "**********")
    #
    # result_summary_dir = "../result/result_summary.csv"
    # try:
    #     result_summary = pd.read_csv(result_summary_dir)
    # except FileNotFoundError:
    #     result_summary = pd.DataFrame(
    #         columns=['model', 'bf_c2', 'bf_r2', 'bf_rc2', 'bf_sum', 'b_c2', 'b_r2', 'b_rc2', 'b_sum'])
    #
    # result_summary.loc[len(result_summary)] = [filename] + count_list
    # result_summary.to_csv(result_summary_dir, index=False)


import logging
import os
import argparse
import time
import random
import numpy as np
from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader

from gurobipy import *
from CGEnv import CGEnv, Transition, ReplayMemory
from models import MLP, GNN, PDN
from params import *
from logger import save_log_file
save_log_file(__file__.split('\\')[-1].split('.')[0])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TARGET_UPDATE_ITER = 2000    # 200, 500, 1000

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def get_config(args=None):
    parser = argparse.ArgumentParser(description="RLCG for VCSP")
    parser.add_argument('--instance_time_limit', type=int, default=600, help="")
    parser.add_argument('--net_type', type=str, default="MLP", choices=["MLP", "GNN"])
    parser.add_argument('--save_dir', type=str, default="../test")
    parser.add_argument('--run_mode', type=str, default="train", choices=["debug", "test", "train"])

    parser.add_argument('--seed', type=int, default=523, help="random seed")
    parser.add_argument('--alpha', type=int, default=100, help="")
    parser.add_argument('--epsilon', type=float, default=0.05, help="")
    parser.add_argument('--gamma', type=float, default=0.95, help="")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="")
    parser.add_argument('--lr_gamma', type=float, default=0.98, help="")
    parser.add_argument('--batch_size', type=int, default=64, help="")
    parser.add_argument('--buffer_size', type=int, default=100000, help="")

    parser.add_argument('--num_episodes', type=int, default=1000, help="number of training episode")
    parser.add_argument('--n_min', type=int, default=50, help="min customers of training instance")
    parser.add_argument('--n_max', type=int, default=75, help="max customers of training instance")
    parser.add_argument('--grad_clamp', type=str_to_bool, default=False, help="")
    parser.add_argument('--load_model', type=str_to_bool, default=False, help="")
    parser.add_argument('--load_case', type=str, default=None, help="")
    parser.add_argument('--print_info', type=str_to_bool, default=False, help="")

    _config = parser.parse_args(args)
    return _config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 不知道有啥区别
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def find_memory_and_model(case, folder):
    file_list = os.listdir(folder)
    memory_file_name, model_file_name = None, None
    if case is not None:
        for file_name in file_list:
            if case in file_name and "memory" in file_name:
                memory_file_name = os.path.join(folder, file_name)
            elif case in file_name and "model" in file_name:
                model_file_name = os.path.join(folder, file_name)
    return memory_file_name, model_file_name

def set_optimizer(model):
    # optimizer = optim.RMSprop(policy_net.parameters(), lr=config.learning_rate)
    # optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    # trainable = filter(lambda x: x.requires_grad, policy_net.parameters())  # 过滤出来需要训练的参数
    # optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    _optimizer = optim.Adam(optimizer_grouped_parameters,
                         lr=config.learning_rate,)
                         # warmup=0.1,
                         # t_total=train_steps)
    return _optimizer

def get_net(_net_type):
    if _net_type == "MLP":
        _policy_net = MLP(xdim=num_featrues, ydim=n_actions).to(device)
        _target_net = MLP(xdim=num_featrues, ydim=n_actions).to(device)
    elif _net_type == "PDN":
        _policy_net = PDN(in_channels=num_node_featrues, out_channels=n_actions, edge_dim=num_edge_featrues).to(device)
        _target_net = PDN(in_channels=num_node_featrues, out_channels=n_actions, edge_dim=num_edge_featrues).to(device)
    elif _net_type == "GNN":
        _policy_net = GNN(in_channels=num_node_featrues, out_channels=n_actions).to(device)
        _target_net = GNN(in_channels=num_node_featrues, out_channels=n_actions).to(device)
    else:
        _policy_net = _target_net = None
    return _policy_net, _target_net

def select_action(_state):
    EPS_END = config.epsilon  # 0.05
    EPS_START = 0.9
    EPS_DECAY = 200     # (0.9 - 0.05) * exp(-1.* 1000 / 200) = 0.005
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            if net_type == "MLP":
                out = policy_net(_state.unsqueeze(0)).squeeze()
            else:
                out = policy_net(_state).squeeze()
            if run_mode == 'debug':
                print("out.max - out.min = ", out.max() - out.min())
            return out.max(0)[1].view(1)
    else:
        return torch.tensor([1], device=device, dtype=torch.long)   # BestEdges2
        # return torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long)

def optimize_model():
    BATCH_SIZE = config.batch_size  # 128
    policy_net.train()
    if len(memory) < BATCH_SIZE:
        return 0, 0
    # elif len(memory) < BATCH_SIZE:
    #     transitions = memory.sample(len(memory))
    else:
        transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    next_states_loader = DataLoader(list(batch.next_state), batch_size=len(batch.state))
    next_states = next(iter(next_states_loader))
    state_loader = DataLoader(list(batch.state), batch_size=len(batch.state))
    state_batch = next(iter(state_loader))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    # policy net 输出为 Qtable，batchsize * n_actions, gather之后得到的是每个选出的action的Q value
    batch_out = policy_net(state_batch)
    state_action_values = batch_out.gather(1, action_batch.unsqueeze(1))
    max_q_value = state_action_values.max()
    # if run_mode == 'debug':
    #     print((batch_out.std(dim=0) / batch_out.mean(dim=0)).detach())
        # print(batch_out.detach())

    next_state_values = target_net(next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (1 - done_batch) * (next_state_values * config.gamma) + reward_batch

    criterion = nn.MSELoss()
    # min (Q(s, a) - [r + (1 - done)Q(s',a')]) ^ 2
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    if config.grad_clamp:
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # print(optimizer.param_groups[0]['lr'])
    return loss.item(), max_q_value.item()

config = get_config()
net_type = config.net_type
save_dir = config.save_dir
run_mode = config.run_mode
set_seed(config.seed)
print(config.__str__())

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "running_logs"), exist_ok=True)
start_time = time.strftime("%H.%M-%m.%d", time.localtime())
test_case = f"a={config.alpha}_e={config.epsilon}_g={config.gamma}_lr={config.learning_rate}_seed={config.seed}_{start_time}"

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

n_actions = len(action_space)
env = CGEnv(device, config, action_space)

policy_net, target_net = get_net(net_type)
optimizer = set_optimizer(policy_net)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
memory = ReplayMemory(config.buffer_size)
start_episode = 0
steps_done = 0  # total steps

# config.load_model = True; config.load_case = "a=100_e=0.05_g=0.99_lr=0.1_seed=123_16.26-10.24"
memory_dir, model_dir = find_memory_and_model(config.load_case, save_dir)
if config.load_model and memory_dir and model_dir:
    memory = ReplayMemory(config.buffer_size, load_dir=memory_dir)
    checkpoint = torch.load(model_dir)
    policy_net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_episode = checkpoint['i_episode']
    steps_done = checkpoint['steps_done']

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if run_mode in ['train', 'test']:
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_logs', test_case))

num_episodes = config.num_episodes
# num_episodes_each_type = round(num_episodes / 3)
# list_episodes = ['C1'] * num_episodes_each_type
# list_episodes.extend(['R1'] * num_episodes_each_type)
# list_episodes.extend(['RC1'] * (num_episodes - 2*num_episodes_each_type))
# random.shuffle(list_episodes)
start = time.time()

for i_episode in range(start_episode, num_episodes):
    ep_reward, ep_loss, ep_max_q = 0, 0, 0
    ep_start = time.time()
    # Initialize the environment and state
    # instance_type = list_episodes[i_episode]
    state = env.reset(run_mode=run_mode)
    for t in count():
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        ep_reward += reward

        if run_mode == 'debug':
            print('ep:{}, t:{}, reward:{:.2f}, done:{}, objval:{:.2f}, heuristic:{}'.format(
                i_episode+1, t+1, reward, done, info['objval'], info['pricing_strategy'],
                ))

        reward = torch.tensor([reward], device=device)
        # Perform one step of the optimization (on the policy network)
        loss, max_q = optimize_model()
        ep_loss += loss
        ep_max_q = max(ep_max_q, max_q)

        memory.push(state, action, next_state, reward, torch.tensor([int(done)], device=device))
        if done:
            # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
            if len(memory) >= config.batch_size:
                scheduler.step()

            print("Episode {:3d}/{}, n: {:2d}, iter: {:3d}, RMP_time: {:.2f}, SP_time: {:.2f}, "
                  "reward: {:.3f}, loss: {:.3f}, driver: {:2d}, objval: {:.2f}, gap: {:.4f}".format(
                i_episode + 1, num_episodes, env.num_customer, t + 1, sum(env.RMP_times), sum(env.SP_times),
                ep_reward, ep_loss, env.num_driver, env.objval, info["gap"]))
            with open (os.path.join(save_dir, f'running_logs/{test_case}.txt'), 'a') as file:
                file.write("Episode {:3d}/{}, n: {:2d}, iter: {:3d}, RMP_time: {:.2f}, SP_time: {:.2f}, "
                           "reward: {:.3f}, loss: {:.3f}, driver: {:2d}, objval: {:.2f}, gap: {:.4f}\n".format(
                    i_episode + 1, num_episodes, env.num_customer, t + 1, sum(env.RMP_times), sum(env.SP_times),
                    ep_reward, ep_loss, env.num_driver, env.objval, info["gap"]))

            if run_mode in ['train', 'test']:
                writer.add_scalar('1_gap', info['gap'], i_episode)
                writer.add_scalar('2_episode_rewards', ep_reward, i_episode)
                writer.add_scalar('3_max_q', max_q, i_episode)
                writer.add_scalar('4_driver', env.num_driver, i_episode)
                writer.add_scalar('5_objval', env.objval, i_episode)
                writer.add_scalar('7_episode_losses', ep_loss, i_episode)
                writer.add_scalar('7_iters', t+1, i_episode)
            break

        # done = torch.tensor([int(done)], device=device)
        # memory.push(state, action, next_state, reward, done)
        state = next_state

        if steps_done % TARGET_UPDATE_ITER == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

    if (i_episode+1) % 200 == 0:
        memory.save(os.path.join(save_dir, f'memory_{test_case}.pkl'))
        save_data = {
            'net':policy_net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'i_episode':i_episode+1,
            'steps_done':steps_done+1,
        }
        torch.save(save_data, os.path.join(save_dir, f'model_{test_case}.pkl'))
        # torch.save(policy_net, os.path.join(save_dir, f'model_{test_case}.pkl'))

if run_mode in ['train', 'test']:
    writer.close()
# with open (os.path.join(save_dir, 'result.txt'), 'a') as file:
#     file.write("test_case: {}, time: {} \n".format(test_case, time.time()-start))

print('Complete')

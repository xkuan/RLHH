import argparse
# from bidict import bidict


""" global """
num_featrues = 14
num_node_featrues = 5
num_edge_featrues = 3
map_edge_heuristics = None


action_space = [
    # "Exact",
    "BestEdges1",
    "BestEdges2",
    "BestEdges3",
    "BestNodes",
    "BestPaths",
]
heuristics = action_space + ["Hyper"]

state_mean = [0.95567846, 1.4837236 , 0.1958487 , 0.30734837, 0.37013346,
              0.78455746, 0.597497  , 0.6845213 , 0.69543827, 0.6841564 ,
              0.35097876, 0.37508938, 0.33730385, 0.03788296]
state_std = [0.06044082, 0.20721085, 0.14330077, 0.2406444 , 0.05192028,
             0.01829603, 0.01080656, 0.00560187, 0.01209558, 0.00559653,
             0.02418068, 0.0264779 , 0.02734187, 0.00158831]

config_test = argparse.ArgumentParser()
config_test.instance_time_limit = 600
config_test.n_min=40
config_test.n_max=60
# config_test.save_dir=None
# config_test.instance_size="small"
config_test.alpha=100
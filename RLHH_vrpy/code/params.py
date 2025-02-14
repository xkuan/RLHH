import argparse
# from bidict import bidict


""" DQN_train """
# net_type = "GNN"
# save_dir = './test' #
# run_mode = 'test'


""" global """
num_featrues = 14
num_node_featrues = 5
num_edge_featrues = 3
probNum = {
    'c1': 9, 'c2': 8,
    'r1': 12, 'r2': 11,
    'rc1': 8, 'rc2': 8,
}
map_edge_heuristics = None
# bidict({
#     ('C', 'threshold'): 'BestEdgesC1',
#     ('C', 'ratio'): 'BestEdgesC2',
#     ('C', 'probability'): 'BestEdgesC3',
#     ('P', 'threshold'): 'BestEdgesP1',
#     ('P', 'ratio'): 'BestEdgesP2',
#     ('P', 'probability'): 'BestEdgesP3',
#     ('CP', 'threshold'): 'BestEdgesCP1',
#     ('CP', 'ratio'): 'BestEdgesCP2',
#     ('CP', 'probability'): 'BestEdgesCP3',
# })

action_space = [
    # "Exact",
    "BestEdges1",
    "BestEdges2",
    "BestEdges3",
    "BestNodes",
    "BestPaths",
]
heuristics = action_space + ["Hyper"]

config_test = argparse.ArgumentParser()
config_test.instance_time_limit = 600
config_test.net_type="MLP"
# config_test.save_dir=None
# config_test.instance_size="small"
config_test.alpha=100
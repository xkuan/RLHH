import math
import pathlib

import numpy as np
import pandas as pd
import networkx as nx

class SolomonDataSet:
    """Reads a Solomon instance and stores the network as DiGraph.

    Args:
        path (pathlib.Path) : Path of Solomon instance to read.
        n_vertices (int, optional):
            Only first n_vertices are read.
            Defaults to None.
    """

    def __init__(self, path: str=None, n_vertices: int=None, preprocess: bool=False,
                 cut_tw: bool=False, threshold: int=None, printData: bool=False,
    ):
        self.G = nx.DiGraph()
        self.printData = printData
        self.max_load = None
        self.instance_name = path.split('/')[-1]
        self.num_customer = n_vertices
        path = pathlib.Path(path)
        self._load(path, n_vertices, preprocess)
        if cut_tw:
            if threshold is None:
                threshold = 90
            self._cut_time_window_span(threshold=threshold)

    def _cut_time_window_span(self, threshold):
        for u in self.G.nodes(data=True):
            if u[0] in ["Source", "Sink"]:
                if self.printData:
                    print(u)
                continue
            if u[1]["tw_span"] > threshold:
                u[1]["dueTime"] = u[1]["readyTime"] + threshold  # 会更新graph
                u[1]["tw_span"] = threshold
            if self.printData:
                print(u)

    @staticmethod
    def _cal_distance(node1, node2):
        x1, y1 = node1['x'], node1['y']
        x2, y2 = node2['x'], node2['y']
        dist = math.pow((math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2)), 0.5)
        return dist

    def _check_feasibility(self, node_u, node_v):
        travelTime = self._cal_distance(node_u, node_v)
        if node_u['readyTime'] + node_u['serviceTime'] + travelTime > node_v['dueTime']:
            return False
        elif node_u['demand'] + node_v['demand'] > self.max_load:
            return False
        else:
            return True

    def _load(self, path, n_vertices=None, preprocess=False):
        # Read vehicle capacity
        with open(path, 'r') as fp:
            for i, line in enumerate(fp):
                if i == 4:
                    self.vehicleNum = int(line.split()[0])
                    self.max_load = int(line.split()[1])
                    break

        # Read nodes from txt file
        df_solomon = pd.read_csv(
            path,
            sep="\s+",
            skip_blank_lines=True,
            skiprows=7,
            nrows=n_vertices + 1,
        )

        # Scan each line of the file and add nodes to the network
        for values in df_solomon.itertuples():
            node_name = "Source" if values[1] == 0 else np.uint32(values[1]).item()
            self.G.add_node(
                node_name,
                x=np.float64(values[2]).item(),
                y=np.float64(values[3]).item(),
                demand=np.uint32(values[4]).item(),
                lower=np.uint32(values[5]).item(),
                upper=np.uint32(values[6]).item(),
                service_time=np.uint32(values[7]).item(),
                tw_span=np.uint32(values[6]).item() - np.uint32(values[5]).item(),
            )
            # Add Sink as copy of Source
            if node_name == "Source":
                self.G.add_node(
                    "Sink",
                    x=np.float64(values[2]).item(),
                    y=np.float64(values[3]).item(),
                    demand=np.uint32(values[4]).item(),
                    lower=np.uint32(values[5]).item(),
                    upper=np.uint32(values[6]).item(),
                    service_time=np.uint32(values[7]).item(),
                    tw_span=np.uint32(values[6]).item() - np.uint32(values[5]).item(),
                )

        # Add the edges
        if preprocess:
            # the graph only contain edges satisify time_window and max_load constraints
            for u in self.G.nodes():
                if u != "Sink":
                    for v in self.G.nodes():
                        if v != "Source" and u != v:
                            # if (u, v) == ("Source", "Sink"):
                            #     continue
                            node_u = self.G.nodes[u]
                            node_v = self.G.nodes[v]
                            if self._check_feasibility(node_u, node_v):
                                dist = round(self._cal_distance(node_u, node_v), 4)
                                self.G.add_edge(u, v, cost=dist, time=dist)
        else:
            # the graph is complete
            for u in self.G.nodes():
                if u != "Sink":
                    for v in self.G.nodes():
                        if v != "Source" and u != v:
                            # if (u, v) == ("Source", "Sink"):
                            #     continue
                            dist = round(self._cal_distance(self.G.nodes[u], self.G.nodes[v]), 4)
                            self.G.add_edge(u, v, cost=dist, time=dist)



if __name__ == "__main__":
    data = SolomonDataSet(path='../data/c101.txt', n_vertices=100)
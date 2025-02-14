import math
import pathlib

import numpy as np
import pandas as pd
import networkx as nx
from generate_data import generate_data


class VCSPDataSet:
    def __init__(self, path: str=None, n_vertices=None):
        self.G = nx.DiGraph()
        if path is not None:
            # self.instance_name = path.split('/')[-1]
            df_shifts = pd.read_csv(pathlib.Path(path))
        elif n_vertices is not None:
            df_shifts = generate_data(n_vertices)
        else:
            raise "Input path or n_vertices!"

        df = df_shifts.reset_index()
        self.shifts = df.values
        self.num_shift = len(df_shifts)
        # All durations are in minutes.
        self.max_driving_time = 480     # 8 hours.
        self.max_driving_time_without_break = 240  # 4 hours
        self.min_break_time = 30
        self.min_delay_between_shifts = 2
        self.max_working_time = 720
        self.min_working_time = 240
        self.sign_on_time = 10          # worked time before a workday start
        self.sign_off_time = 15         # worked time after a workday end
        self.fixed_driver_cost = 1440   # fixed cost of a driver (convert to time cost)

        self.min_start_time: int = 0
        self.max_end_time: int = 0

        self.build_data(df_shifts)

    def _check_feasibility(self, node_u, node_v):
        if node_u['shift_id'] == 'Source':  # start of a workday
            return self.max_end_time - node_v['lower'] > self.min_working_time
        # elif node_v['shift_id'] == 'Sink':    # end of a workday
        #     return node_u['upper'] - self.min_start_time > self.min_working_time
        else:
            return node_u['upper'] + self.min_delay_between_shifts < node_v['lower']

    def _load(self, path):
        """默认输入数据已经按照时间顺序排列好了"""
        # Read shifts from txt file or csv file
        sep = "\s+" if str(path).split('.')[-1] == "txt" else ","
        df_shifts = pd.read_csv(path, sep=sep)
        if isinstance(self.num_shift, int) and self.num_shift < len(df_shifts):
            df_shifts = df_shifts.sample(self.num_shift)
        df_shifts = df_shifts.sort_values(by="start_minute", ascending=True).reset_index(drop=True)

        return df_shifts

    def build_data(self, df_shifts):
        # Computed data.
        shifts = df_shifts.values
        self.min_start_time = min(shift[2] for shift in shifts)
        self.max_end_time = max(shift[3] for shift in shifts)

        # Scan each line of the file and add nodes to the network
        for values in df_shifts.itertuples(index=True):
            self.G.add_node(
                np.uint32(values[0] + 1).item(),  # node_name
                shift_id=np.uint32(values[0] + 1).item(),
                lower=np.uint32(values[3]).item(),
                upper=np.uint32(values[4]).item(),
                duration=np.uint32(values[5]).item(),
            )
        # for node_name in ["Source", "Sink"]:
        self.G.add_node(
            "Source",
            shift_id="Source",
            lower=np.uint32(0).item(),
            upper=np.uint32(0).item(),
            duration=np.uint32(0).item(),
        )
        self.G.add_node(
            "Sink",
            shift_id="Sink",
            lower=np.uint32(1440).item(),
            upper=np.uint32(1440).item(),
            duration=np.uint32(0).item(),
        )

        # Add the edges from source or to sink
        self.G.add_edge('Source', 'Sink',
                        type='virtual',
                        cost=[0],
                        shift_num=0,  # 终点是 shift 则为1
                        working_time=0,
                        driving_time=0,
                        drive_without_break=0,
                        )
        for v in self.G.nodes():
            if v in ["Source", "Sink"] or not self._check_feasibility(self.G.nodes['Source'], self.G.nodes[v]):
                continue
            self.G.add_edge('Source', v,
                            type='sign_on',
                            cost=[self.fixed_driver_cost + self.G.nodes[v]['duration'] + self.sign_on_time],
                            shift_num=1,  # 终点是 shift 则为 1
                            working_time=self.G.nodes[v]['duration'] + self.sign_on_time,
                            driving_time=self.G.nodes[v]['duration'],
                            drive_without_break=1,
                            )
        for v in self.G.nodes():
            if v in ["Source", "Sink"] or not self._check_feasibility(self.G.nodes[v], self.G.nodes['Sink']):
                continue
            self.G.add_edge(v, 'Sink',
                            type='sign_off',
                            cost=[self.sign_off_time],
                            shift_num=0,  # 终点是 shift 则为1
                            working_time=self.sign_off_time,
                            driving_time=0,
                            drive_without_break=1,      # 因为这里驾驶时间是0,所以这个设置为0,1都一样
                            )
        # Add the edges between shifts
        for u in self.G.nodes():
            for v in self.G.nodes():
                if v in ["Source", "Sink"] or u in ["Source", "Sink"]:
                    continue
                node_u = self.G.nodes[u]
                node_v = self.G.nodes[v]
                if not self._check_feasibility(node_u, node_v):
                    continue
                drive_without_break = int(node_v['lower'] - node_u['upper'] < self.min_break_time)
                self.G.add_edge(u, v,
                                type='work',
                                cost=[node_v['upper'] - node_u['upper']],
                                shift_num=1,  # 终点是 shift 则为1
                                working_time=node_v['upper'] - node_u['upper'],
                                driving_time=node_v['duration'],
                                drive_without_break=drive_without_break,
                                )

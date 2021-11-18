import pandas as pd

import numpy as np

import networkx as nx

class ReadGraph():
    def __init__(self, args):
        self.args = args
        #加载rel文件
        self.dataset = self.args.dataset
        self.data_path = './raw_data/' + self.dataset + '/'
        self.rel_file = self.dataset + '.rel'
    #根据rel文件生成有向图
    def _load_graph(self):
        map_info = pd.read_csv(self.data_path + self.rel_file)
        start = np.array(map_info['origin_id'])
        end = np.array(map_info['destination_id'])
        nG = nx.DiGraph()
        for i in range(len(start)):
            nG.add_edge(start[i],end[i])
        return nG





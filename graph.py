from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch


class DataCenter(object):
    def __init__(self, args):
        super(DataCenter, self).__init__()
        self.n = 0
        self.m = 0
        self.dataset = args.dataset
        self.node_emb_size = args.node_emb_size

    def load_dataset(self):
        print("Loading {}.edges ...".format(self.dataset))

        node_map = {}
        adj_lists = defaultdict(set)
        with open("./data/" + self.dataset + ".edges") as f:
            for line in f.readlines():
                u, v = line.strip().split()
                u, v = int(u), int(v)
                if u not in node_map:
                    node_map[u] = self.n
                    self.n += 1
                if v not in node_map:
                    node_map[v] = self.n
                    self.n += 1
                adj_lists[node_map[u]].add(node_map[v])
                adj_lists[node_map[v]].add(node_map[u])
                self.m += 1

        train_indices = self._split_data()

        setattr(self, self.dataset+"_adj_lists", adj_lists)
        setattr(self, self.dataset+"_train", train_indices)
        print("Finish loading {}.edges".format(self.dataset))

    def _split_data(self):
        rand_indices = np.random.permutation(self.n)
        train_indices = rand_indices[:]
        return train_indices
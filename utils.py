import sys
import os
import torch
import random
import math
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


def train(data_center, graph_model, batch_size):
    train_nodes = getattr(data_center, data_center.dataset + "_train")
    train_nodes = shuffle(train_nodes)

    models = [graph_model]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.01)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / batch_size)
    visited_nodes = set()

    for index in range(batches):
        nodes_batch = train_nodes[index * batch_size: (index + 1) * batch_size]
        visited_nodes |= set(nodes_batch)
        embs_batch = graph_model(nodes_batch)

        '''
        Add downstream task, calculate loss
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        '''

        print('Step [{}/{}], Dealed Nodes [{}/{}] '.format(index + 1, batches, len(visited_nodes), len(train_nodes)))

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

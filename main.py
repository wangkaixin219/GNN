from graph import DataCenter
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataset', type=str, default='syn')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--node_emb_size', type=int, default=128)
parser.add_argument('--hidden_emb_size', type=int, default=128)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_center = DataCenter(args)
    data_center.load_dataset()
    adj_lists = getattr(data_center, data_center.dataset+"_adj_lists")

    graph_model = GraphSage(
        n_nodes=data_center.n,
        num_layers=args.num_layers,
        in_size=args.node_emb_size,
        out_size=args.hidden_emb_size,
        adj_lists=adj_lists,
        device=device,
        gcn=args.gcn,
        agg_func=args.agg_func)
    graph_model.to(device)

    train(data_center, graph_model, args.batch_size)


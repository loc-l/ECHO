from torch_geometric.datasets import Reddit, Flickr
from ogb.nodeproppred import PygNodePropPredDataset

from fast_sampler import to_row_major
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

import os
import torch

def get_reddit(path):
    if os.path.exists(f'{path}/processed/echo_graph.pt'):
        data = torch.load(f'{path}/processed/echo_graph.pt')
        return data
    
    dataset = Reddit(path, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.num_features = dataset.num_features
    data.processed_dir = dataset.processed_dir
    data.num_nodes = data.x.shape[0]
    data.x = to_row_major(data.x)
    torch.save(data, f'{data.processed_dir}/echo_graph.pt')
    
    return data


def get_flickr(path):
    if os.path.exists(f'{path}/processed/echo_graph.pt'):
        data = torch.load(f'{path}/processed/echo_graph.pt')
        return data
    
    dataset = Flickr(path, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.num_features = dataset.num_features
    data.processed_dir = dataset.processed_dir
    data.num_nodes = data.x.shape[0]
    data.x = to_row_major(data.x)
    torch.save(data, f'{data.processed_dir}/echo_graph.pt')
    
    return data


def get_ogb_data(name, path):
    if os.path.exists(f'{path}/{name.replace("-", "_")}/processed/echo_graph.pt'):
        data = torch.load(f'{path}/{name.replace("-", "_")}/processed/echo_graph.pt')
        return data
    dataset = PygNodePropPredDataset(name.replace('_', '-'), root=path, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.train_mask = torch.tensor([False]*data.num_nodes)
    data.val_mask = torch.tensor([False]*data.num_nodes)
    data.test_mask = torch.tensor([False]*data.num_nodes)

    split_idx = dataset.get_idx_split()
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']] = True
    data.test_mask[split_idx['test']] = True
    data.processed_dir = dataset.processed_dir
    data.num_features, data.num_classes = dataset.num_features, dataset.num_classes
    data.y = data.y.squeeze(1)
    data.x = to_row_major(data.x)
    torch.save(data, f'{data.processed_dir}/echo_graph.pt')

    return data


def get_dataset(name, path):
    if name.lower()=='flickr':
        return get_flickr(path)
    if name.lower()=='reddit':
        return get_reddit(path)
    if 'ogb' in name.lower():
        return get_ogb_data(name, path)


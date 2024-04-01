from typing import List

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.use_bn = use_bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs):
        if isinstance(adjs, List): 
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    if self.use_bn:
                        x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return torch.log_softmax(x, dim=-1)
        else:
            for i, conv in enumerate(self.convs):
                x = conv(x, adjs)
                if i != len(self.convs) - 1:
                    if self.use_bn:
                        x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    if self.use_bn:
                        x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        return x_all

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Linear(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Linear(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Linear(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        if isinstance(adjs, List):
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return x.log_softmax(dim=-1)
        else:
            for i, conv in enumerate(self.convs):
                x_target = x
                x = conv(x, adjs)
                x = x + self.skips[i](x_target)
                if i != len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []
            for _, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())
                del x
                del edge_index
                del x_target

            x_all = torch.cat(xs, dim=0)
        return x_all


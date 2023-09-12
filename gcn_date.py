import os
import torch
import pandas
import ogb

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch_geometric.utils as U
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

"""
Download datasets
"""

#transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])

ogbndataset = PygNodePropPredDataset(name='ogbn-arxiv', root = 'dataset/')
dataArxiv = ogbndataset[0]  # Get the first graph object.

print()
print(f'Dataset: {ogbndataset}:')
print('======================')

print()
print(dataArxiv)
print('===========================================================================================================')

split_idx = ogbndataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

"""
Add date of publication as default value to feature vector, [old<new]
"""
def dateAsDefault(data):
    a = torch.empty(169343, 129)
    for i in range(len(data.x)):
        a[i] = torch.cat([data.x[i], data.node_year[i]], dim=0)
    data.x = a
    return data

"""
Add date of publication as normalized value sum to 1 to feature vector, min: [old<new], max: [old>new]
"""
def dateNormalized(data, minormax):
    data.node_year = data.node_year.squeeze(1).float()
    if minormax == "min":
        data.node_year = data.node_year - data.node_year.min()
    elif minormax == "max":
        data.node_year = data.node_year - data.node_year.max()
        data.node_year = torch.abs(data.node_year)
    else:
        print("Selected min or max")
        return
    data.node_year.div_(data.node_year.sum(dim=-1, keepdim=True).clamp_(min=1.))
    data.node_year = data.node_year.unsqueeze(1)
    a = torch.empty(169343, 129)
    for i in range(len(data.x)):
        a[i] = torch.cat([data.x[i], data.node_year[i]], dim=0)
    data.x = a
    return data

"""
Add date of publication as normalized value between 0 and 1 to feature vector, min: [old<new], max: [old>new]
"""
def dateNormalizedZeroOne(data, minormax):
    data.node_year = data.node_year.squeeze(1).float()
    if minormax == "min":
        data.node_year = data.node_year - data.node_year.min()
    elif minormax == "max":
        data.node_year = data.node_year - data.node_year.max()
        data.node_year = torch.abs(data.node_year)
    else:
        print("Selected min or max")
        return
    data.node_year.div_(data.node_year.max())
    data.node_year = data.node_year.unsqueeze(1)
    a = torch.empty(169343, 129)
    for i in range(len(data.x)):
        a[i] = torch.cat([data.x[i], data.node_year[i]], dim=0)
    data.x = a
    return data

"""
Add date of publication as max and min values to feature vector, min: [old<new], max: [old>new]
"""
def dateMinMax(data, minormax):
    if minormax == "min":
        data.node_year = data.node_year - data.node_year.min()
    elif minormax == "max":
        data.node_year = data.node_year - data.node_year.max()
        data.node_year = torch.abs(data.node_year)
    else:
        print("Selected min or max")
        return
    a = torch.empty(169343, 129)
    for i in range(len(data.x)):
        a[i] = torch.cat([data.x[i], data.node_year[i]], dim=0)
    data.x = a
    return data

"""
Add date of publication as edge weights. Edge gets weight or source or target
"""
def dateEdge(data, sourceortarget):
    if sourceortarget == "source":
        temp = data.edge_index[0]
    elif sourceortarget == "target":
        temp = data.edge_index[1]
    else:
        print("Selecte source or target")
        return
    edge_weight = torch.ones(dataArxiv.edge_index.size(1))
    for i in range(dataArxiv.edge_index.size(1)):
        edge_weight[i] = data.node_year[temp[i]]
    data.edge_weight = edge_weight
    return data

"""
First add edge_weights to data, since we need to use the edge_index. But for training we use a symmetric adj_t so we lose edge_index.
"""
    
# dataArxivEdgeTarget = dataArxiv.clone()
# dataArxivEdgeTarget = dateEdge(dataArxivEdgeTarget, "target")
# print()
# print(dataArxivEdgeTarget)
# print('===========================================================================================================')

# dataArxivEdgeSource = dataArxiv.clone()
# dataArxivEdgeSource = dateEdge(dataArxivEdgeSource, "source")
# print()
# print(dataArxivEdgeSource)
# print('===========================================================================================================')

"""
Do the required transformations
"""
transform=T.ToSparseTensor()
transform(dataArxiv)
# transform(dataArxivEdgeTarget)
# transform(dataArxivEdgeSource)
dataArxiv.adj_t = dataArxiv.adj_t.to_symmetric()
# dataArxivEdgeTarget.adj_t = dataArxivEdgeTarget.adj_t.to_symmetric()
# dataArxivEdgeSource.adj_t = dataArxivEdgeSource.adj_t.to_symmetric()
        
# dataArxivDateAsDefault = dataArxiv.clone()
# dataArxivDateAsDefault = dateAsDefault(dataArxivDateAsDefault)
# print()
# print(dataArxivDateAsDefault)
# print('===========================================================================================================')

dataArxivDateNormalizedMin = dataArxiv.clone()
dataArxivDateNormalizedMin = dateNormalized(dataArxivDateNormalizedMin, "min")
print()
print(dataArxivDateNormalizedMin)
print(dataArxivDateNormalizedMin.node_year)
print(sum(dataArxivDateNormalizedMin.node_year))
print('===========================================================================================================')

dataArxivDateNormalizedMax = dataArxiv.clone()
dataArxivDateNormalizedMax = dateNormalized(dataArxivDateNormalizedMax, "max")
print()
print(dataArxivDateNormalizedMax)
print(dataArxivDateNormalizedMax.node_year)
print(sum(dataArxivDateNormalizedMax.node_year))
print('===========================================================================================================')

# dataArxivDateMinMax = dataArxiv.clone()
# dataArxivDateMinMax = dateMinMax(dataArxivDateMinMax, "min")
# print()
# print(dataArxivDateMinMax)
# print('===========================================================================================================')

# dataArxivDateMaxMin = dataArxiv.clone()
# dataArxivDateMaxMin = dateMinMax(dataArxivDateMaxMin, "max")
# print()
# print(dataArxivDateMaxMin)
# print('===========================================================================================================')

dataArxivDateNormalizedZeroOne = dataArxiv.clone()
dataArxivDateNormalizedZeroOne = dateNormalizedZeroOne(dataArxivDateNormalizedZeroOne, "min")
print()
print(dataArxivDateNormalizedZeroOne)
print('===========================================================================================================')

dataArxivDateNormalizedOneZero = dataArxiv.clone()
dataArxivDateNormalizedOneZero = dateNormalizedZeroOne(dataArxivDateNormalizedOneZero, "max")
print()
print(dataArxivDateNormalizedOneZero)
print('===========================================================================================================')

"""
WandB
"""

enable_wandb = True
if enable_wandb:
    import wandb
    wandb.login()
    
import pandas
def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pandas.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})
    
sweep_config = {
    "name": "gcn-test-date-5-6-model-5-sub-models-3-4-runs-10",
    "method": "grid",
    "parameters": {
        "hidden_channels": {
            "value": 128
        },
        "weight_decay": {
            "value": 0,
        },
        "lr": {
            "value": 5e-3
        },
        "epochs": {
            "value": 500
        },
        "dropout": {
            "value": 0.3
        },
        "skip_coefficient": {
            "value": 1
        },
        "num_layers": {
            "value": 10
        },
        "model": {
            "value": 5
        },
        "date_id": {
            "values": [5,6]
        },
        "sub_model":{
            "values": [3,4]
        }
    }
}

#sweep_id_cora = wandb.sweep(sweep_config, project="GCN_testing_Cora")
#sweep_id_citeseer = wandb.sweep(sweep_config, project="GCN_testing_CiteSeer")
#sweep_id_pubmed = wandb.sweep(sweep_config, project="GCN_testing_Pubmed")
#sweep_id_arxiv = wandb.sweep(sweep_config, project="GCN_testing_Arxiv")

sweep_config = {
    "name": "gcn-test-mem",
    "method": "grid",
    "parameters": {
        "hidden_channels": {
            "value": 128
        },
        "weight_decay": {
            "value": 0,
        },
        "lr": {
            "value": 5e-3
        },
        "epochs": {
            "value": 500
        },
        "dropout": {
            "value": 0.3
        },
        "skip_coefficient": {
            "value": 1
        },
        "num_layers": {
            "value": 10
        },
        "model": {
            "value": 5
        },
        "date_id": {
            "value": 6
        },
        "sub_model":{
            "value": 3
        }
    }
}

sweep_id_test = wandb.sweep(sweep_config, project="uncategorized")

"""
GCN_Default
"""
from torch_geometric.nn import GCNConv

class GCN_Default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        #torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
"""
GCN Multilayer
"""
class GCN_Multilayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

"""
GCN Skip 1 layer
"""
class GCN_skip_1_layer_default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = x + (self.skip_coefficient * x_skip)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_skip = x
        x = self.convs[-1](x, edge_index)
        return x
    
"""
GCN Skip 2 layer
"""
class GCN_skip_2_layer_default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i%2 == 0:
                x = x + (self.skip_coefficient * x_skip)
                x_skip = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
        
"""
GCN Skip 2 layer linear_in_out
"""
class GCN_skip_2_layer_linear_in_out(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i%2 == 0:
                x = x + (self.skip_coefficient * x_skip)
                #x_skip = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out.append(x)
        x = sum(out)
        x = self.convs[-1](x)
        return x
        
"""
GCN Skip 2 layer pre-activation
"""
class GCN_skip_2_layer_pre_activation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.bns[i-1](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout , training=self.training)
            x = self.convs[i](x, edge_index)
            if i%2 == 0:
                x = x + (self.skip_coefficient * x_skip)
                x_skip = x
        x = self.convs[-1](x, edge_index)
        return x

"""
GCN Skip 2 layer linear_in_out add bns,relu,dropout after linear
"""
class GCN_skip_2_layer_linear_in_out_3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i%2 == 0:
                x = x + (self.skip_coefficient * x_skip)
                #x_skip = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out.append(x)
        x = sum(out)
        x = self.convs[-1](x)
        return x

"""
GCN Skip 2 layer linear_in_out default, every layer
"""
class GCN_skip_2_layer_linear_in_out_4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = x + (self.skip_coefficient * x_skip)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out.append(x)
        x = sum(out)
        x = self.convs[-1](x)
        return x

"""
GCN Skip 2 layer linear_in_out default, every layer, plus bns, relu, dropout
"""
class GCN_skip_2_layer_linear_in_out_5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient):
        super().__init__()
        #torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers -2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = x + (self.skip_coefficient * x_skip)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out.append(x)
        x = sum(out)
        x = self.convs[-1](x)
        return x


"""
Logger
"""
class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Argmax {argmax}')
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                end_test = r[:, 2].max().item()
                best_results.append((train1, valid, train2, test, end_test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gcn/highest-train-mean"] = r.mean()
            wandb.summary["gcn/highest-train-std"] = r.std()
            wandb.log({"gcn/highest-train-mean": r.mean(), "gcn/highest-train-std": r.std()})
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gcn/highest-valid-mean"] = r.mean()
            wandb.summary["gcn/highest-valid-std"] = r.std()
            wandb.log({"gcn/highest-valid-mean": r.mean(), "gcn/highest-valid-std": r.std()})
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gcn/final-train-mean"] = r.mean()
            wandb.summary["gcn/final-train-std"] = r.std()
            wandb.log({"gcn/final-train-mean": r.mean(), "gcn/final-train-std": r.std()})
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gcn/final-test-mean"] = r.mean()
            wandb.summary["gcn/final-test-std"] = r.std()
            wandb.log({"gcn/final-test-mean": r.mean(), "gcn/final-test-std": r.std()})
            r = best_result[:, 4]
            print(f' Highest Test: {r.max():.2f} ± {r.std():.2f}')
            wandb.summary["gcn/highest-test-max"] = r.max()
            wandb.summary["gcn/highest-test-std"] = r.std()
            wandb.log({"gcn/highest-test-max": r.max(), "gcn/highest-test-std": r.std()})

"""
Train/Test
"""
def runPlanetoid(data, model, logger):

    def train(model, data, optimizer, criterion):
          model.train()
          optimizer.zero_grad()  # Clear gradients.
          out = model(data.x, data.edge_index)  # Perform a single forward pass.
          loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          return loss

    def test(model, data):
          model.eval()
          out = model(data.x, data.edge_index)
          pred = out.argmax(dim=1)  # Use the class with highest probability.
          
          train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
          train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
          
          valid_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
          val_acc = int(valid_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
          
          test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
          test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
          
          return train_acc, val_acc, test_acc

    for run in range(1):
        if enable_wandb:
            wandb.watch(model)
            summary = dict()
            wandb.summary = summary
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, wandb.config.epochs+1):
            loss = train(model, data, optimizer, criterion)
            result = test(model, data)
            logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()
    
    if enable_wandb:
        wandb.finish()

def runOGB(data, model, logger):
    evaluator = Evaluator(name='ogbn-arxiv')

    def train(model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = criterion(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        return loss

    def test(model, data, evaluator):
        model.eval()
        out = model(data.x, data.adj_t)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

        return train_acc, valid_acc, test_acc
    
    for run in range(10):
        if enable_wandb:
            wandb.watch(model)
            summary = dict()
            wandb.summary = summary
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, wandb.config.epochs+1):
            loss = train(model, data, optimizer, criterion)
            wandb.log({"gcn/loss": loss})
            result = test(model, data, evaluator)
            logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()
    
    if enable_wandb:
        wandb.finish()
device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)

def runCora():
    wandb.init()
    global dataCora
    #dataCora.adj_t = dataCora.adj_t.to_symmetric()
    dataCora = dataCora.to(device)
    i = wandb.config.model
    if i == 0:
        model = GCN_Default(planetoiddatasetCora.num_features, wandb.config.hidden_channels, planetoiddatasetCora.num_classes, wandb.config.dropout).to(device)
    if i == 1:
        model = GCN_Multilayer(planetoiddatasetCora.num_features, wandb.config.hidden_channels, planetoiddatasetCora.num_classes, wandb.config.dropout, wandb.config.num_layers).to(device)
    if i == 2:
        model = GCN_skip_1_layer_default(planetoiddatasetCora.num_features, wandb.config.hidden_channels, planetoiddatasetCora.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 3:
        model = GCN_skip_2_layer_default(planetoiddatasetCora.num_features, wandb.config.hidden_channels, planetoiddatasetCora.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 4:
        model = GCN_skip_2_layer_pre_activation(planetoiddatasetCora.num_features, wandb.config.hidden_channels, planetoiddatasetCora.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    logger = Logger(1)
    runPlanetoid(dataCora, model, logger)
    print(model)
    print(dataCora)

#wandb.agent(sweep_id_test, project="uncategorized", function=runCora)
#wandb.agent(sweep_id_cora, project="GCN_testing_Cora", function=runCora)

def runCiteseer():
    wandb.init()
    global dataCiteSeer
    #dataCora.adj_t = dataCora.adj_t.to_symmetric()
    dataCiteSeer = dataCiteSeer.to(device)
    i = wandb.config.model
    if i == 0:
        model = GCN_Default(planatoiddatasetCiteSeer.num_features, wandb.config.hidden_channels, planatoiddatasetCiteSeer.num_classes, wandb.config.dropout).to(device)
    if i == 1:
        model = GCN_Multilayer(planatoiddatasetCiteSeer.num_features, wandb.config.hidden_channels, planatoiddatasetCiteSeer.num_classes, wandb.config.dropout, wandb.config.num_layers).to(device)
    if i == 2:
        model = GCN_skip_1_layer_default(planatoiddatasetCiteSeer.num_features, wandb.config.hidden_channels, planatoiddatasetCiteSeer.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 3:
        model = GCN_skip_2_layer_default(planatoiddatasetCiteSeer.num_features, wandb.config.hidden_channels, planatoiddatasetCiteSeer.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 4:
        model = GCN_skip_2_layer_pre_activation(planatoiddatasetCiteSeer.num_features, wandb.config.hidden_channels, planatoiddatasetCiteSeer.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    logger = Logger(1)
    runPlanetoid(dataCiteSeer, model, logger)
    print(model)
    print(dataCiteSeer)

#wandb.agent(sweep_id_citeseer, project="GCN_testing_CiteSeer", function=runCiteseer)

def runPubmed():
    wandb.init()
    global dataPubMed
    #dataCora.adj_t = dataCora.adj_t.to_symmetric()
    dataPubMed = dataPubMed.to(device)
    i = wandb.config.model
    if i == 0:
        model = GCN_Default(planatoiddatasetPubMed.num_features, wandb.config.hidden_channels, planatoiddatasetPubMed.num_classes, wandb.config.dropout).to(device)
    if i == 1:
        model = GCN_Multilayer(planatoiddatasetPubMed.num_features, wandb.config.hidden_channels, planatoiddatasetPubMed.num_classes, wandb.config.dropout, wandb.config.num_layers).to(device)
    if i == 2:
        model = GCN_skip_1_layer_default(planatoiddatasetPubMed.num_features, wandb.config.hidden_channels, planatoiddatasetPubMed.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 3:
        model = GCN_skip_2_layer_default(planatoiddatasetPubMed.num_features, wandb.config.hidden_channels, planatoiddatasetPubMed.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 4:
        model = GCN_skip_2_layer_pre_activation(planatoiddatasetPubMed.num_features, wandb.config.hidden_channels, planatoiddatasetPubMed.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    logger = Logger(1)
    runPlanetoid(dataPubMed, model, logger)
    print(model)
    print(dataPubMed)

#wandb.agent(sweep_id_pubmed, project="GCN_testing_Pubmed", function=runPubmed)
    
#dataArxiv.adj_t = dataArxiv.adj_t.to_symmetric()
#dataArxiv = dataArxiv.to(device)

def runArxiv():
    wandb.init()
    i = wandb.config.model
    if i == 0:
        model = GCN_Default(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout).to(device)
    if i == 1:
        model = GCN_Multilayer(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers).to(device)
    if i == 2:
        model = GCN_skip_1_layer_default(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 3:
        model = GCN_skip_2_layer_default(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 4:
        model = GCN_skip_2_layer_pre_activation(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 5:
        model = GCN_skip_2_layer_linear_in_out(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 6:
        model = GCN_skip_2_layer_sum_all(ogbndataset.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    logger = Logger(10)
    runOGB(dataArxivWeight, model, logger)
    print(model)

#wandb.agent(sweep_id_test, project="uncategorized", function=runArxiv)
#wandb.agent(sweep_id_arxiv, project="GCN_testing_Arxiv", function=runArxiv)

"""
Run all Date models
"""
def runArxivDate():
    wandb.init()     
    j = wandb.config.date_id
    if j == 0:
        print("Start dataArxiv")
        data = dataArxiv
    if j == 1:
        print("Start dataArxivDateAsDefault")
        data = dataArxivDateAsDefault
    if j == 2:
        print("Start dataArxivDateMaxMin")
        data = dataArxivDateMaxMin
    if j == 3:
        print("Start dataArxivDateMinMax")
        data = dataArxivDateMinMax
    if j == 4:
        print("Start dataArxivDateNormalizedMax")
        data = dataArxivDateNormalizedMax
    if j == 5:
        print("Start dataArxivDateNormalizedMin")
        data = dataArxivDateNormalizedMin
    if j == 6:
        print("Start dataArxivDateNormalizedOneZero")
        data = dataArxivDateNormalizedOneZero
    if j == 7:
        print("Start dataArxivDateNormalizedZeroOne")
        data = dataArxivDateNormalizedZeroOne
    if j == 8:
        print("Start dataArxivEdgeSource")
        data = dataArxivEdgeSource
    if j == 9:
        print("Start dataArxivEdgeTarget")
        data = dataArxivEdgeTarget
    print()
    print(data)
    print('===========================================================================================================')
    data = data.to(device)
    i = wandb.config.sub_model
    if i == 0:
        print("Sub_model 0")
        model = GCN_skip_2_layer_linear_in_out(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 1:
        print("Sub_model 1")
        model = GCN_skip_2_layer_linear_in_out_1(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 2:
        print("Sub_model 2")
        model = GCN_skip_2_layer_linear_in_out_2(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 3:
        print("Sub_model 3")
        model = GCN_skip_2_layer_linear_in_out_3(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 4:
        print("Sub_model 4")
        model = GCN_skip_2_layer_linear_in_out_4(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    if i == 5:
        print("Sub_model 5")
        model = GCN_skip_2_layer_linear_in_out_5(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient).to(device)
    logger = Logger(10)
    print(sum(p.numel() for p in model.parameters()))
    #runOGB(data, model, logger)
    print(model)
    torch.cuda.empty_cache()

wandb.agent(sweep_id_test, project="uncategorized", function=runArxivDate)
#wandb.agent(sweep_id_arxiv, project="GCN_testing_Arxiv", function=runArxivDate)
#wandb.agent(sweep_id="ri6n8ejx", project="GCN_testing_Arxiv", function=runArxivDate)


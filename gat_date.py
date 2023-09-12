import os
import torch
import pandas
import ogb

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

"""
Download datasets
"""

transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])

planetoiddatasetCora = Planetoid(root='data/Planetoid', name='Cora', transform=transform)

print()
print(f'Dataset: {planetoiddatasetCora}:')
print('======================')
print(f'Number of graphs: {len(planetoiddatasetCora)}')
print(f'Number of features: {planetoiddatasetCora.num_features}')
print(f'Number of classes: {planetoiddatasetCora.num_classes}')

dataCora = planetoiddatasetCora[0]  # Get the first graph object.

print()
print(dataCora)
print('===========================================================================================================')

planatoiddatasetCiteSeer = Planetoid(root='data/Planatoid', name='CiteSeer', transform=transform)

print()
print(f'Dataset: {planatoiddatasetCiteSeer}:')
print('======================')
print(f'Number of graphs: {len(planatoiddatasetCiteSeer)}')
print(f'Number of features: {planatoiddatasetCiteSeer.num_features}')
print(f'Number of classes: {planatoiddatasetCiteSeer.num_classes}')

dataCiteSeer = planatoiddatasetCiteSeer[0]  # Get the first graph object.

print()
print(dataCiteSeer)
print('===========================================================================================================')

planatoiddatasetPubMed = Planetoid(root='data/Planatoid', name='PubMed', transform=transform)

print()
print(f'Dataset: {planatoiddatasetPubMed}:')
print('======================')
print(f'Number of graphs: {len(planatoiddatasetPubMed)}')
print(f'Number of features: {planatoiddatasetPubMed.num_features}')
print(f'Number of classes: {planatoiddatasetPubMed.num_classes}')

dataPubMed = planatoiddatasetPubMed[0]  # Get the first graph object.

print()
print(dataPubMed)
print('===========================================================================================================')

ogbndataset = PygNodePropPredDataset(name='ogbn-arxiv', root = 'dataset/')
dataArxiv = ogbndataset[0]  # Get the first graph object.

print()
print(f'Dataset: {ogbndataset}:')
print('======================')

print()
print(dataArxiv)
print('===========================================================================================================')

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
        
dataArxivDateAsDefault = dataArxiv.clone()
dataArxivDateAsDefault = dateAsDefault(dataArxivDateAsDefault)
print()
print(dataArxivDateAsDefault)
print('===========================================================================================================')

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

dataArxivDateMinMax = dataArxiv.clone()
dataArxivDateMinMax = dateMinMax(dataArxivDateMinMax, "min")
print()
print(dataArxivDateMinMax)
print('===========================================================================================================')

dataArxivDateMaxMin = dataArxiv.clone()
dataArxivDateMaxMin = dateMinMax(dataArxivDateMaxMin, "max")
print()
print(dataArxivDateMaxMin)
print('===========================================================================================================')

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

split_idx = ogbndataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

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
    "name": "gat-test-date-6-model-5-in-head-8",
    "method": "grid",
    "parameters": {
        "hidden_channels": {
            "value": 16
        },
        "weight_decay": {
            "value": 0
        },
        "lr": {
            "value": 1e-2
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
            "value": 6
        },
        "model": {
            "value": 5
        },
        "in_head": {
            "value": 8
        },
        "out_head": {
            "value": 1
        },
        "date_id": {
            "value": 6
        }
    }
}

#Register the Sweep with W&B

#sweep_id_cora = wandb.sweep(sweep_config, project="GAT_testing_Cora")
#sweep_id_citeseer = wandb.sweep(sweep_config, project="GAT_testing_CiteSeer")
#sweep_id_pubmed = wandb.sweep(sweep_config, project="GAT_testing_Pubmed")
sweep_id_arxiv = wandb.sweep(sweep_config, project="GAT_testing_Arxiv")

sweep_config = {
    "name": "gat-test",
    "method": "grid",
    "parameters": {
        "hidden_channels": {
            "value": 8
        },
        "weight_decay": {
            "value": 0
        },
        "lr": {
            "value": 1e-2
        },
        "epochs": {
            "value": 100
        },
        "dropout": {
            "value": 0.6
        },
        "skip_coefficient": {
            "value": 1
        },
        "num_layers": {
            "value": 3
        },
        "model": {
            "value": 1
        },
        "in_head": {
            "value": 8
        },
        "out_head": {
            "value": 1
        }
    }
}

#sweep_id_test = wandb.sweep(sweep_config, project="uncategorized")

"""
GAT_Default
"""


from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT_Default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout)
        self.conv2 = GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
"""
GAT Multilayer
"""

class GAT_Multilayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout))
        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

"""
GAT Skip 1 layer
"""

class GAT_skip_1_layer_default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[0](x, edge_index)
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
            x_skip = x
        x = self.convs[-1](x, edge_index)
        return x
    
"""
GAT Skip 2 layer
"""
class GAT_skip_2_layer_default(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
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
GAT Skip 2 layer linear_in_out
"""

class GAT_skip_2_layer_linear_in_out(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels*in_head))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(torch.nn.Linear(hidden_channels*in_head, out_channels))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        #x = F.dropout(x, p=self.dropout, training=self.training)
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
GAT Skip 2 layer sum all
"""

class GAT_skip_2_layer_sum_all(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x, edge_index)
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
        x = self.convs[-1](x, edge_index)
        return x

"""
GAT Skip 2 layer pre-activation
"""

class GAT_skip_2_layer_pre_activation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, in_head, dropout=self.dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(GATConv(hidden_channels*in_head, out_channels, out_head, dropout=self.dropout))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[0](x, edge_index)
        x_skip = x
        for i in range(1,(len(self.convs)-1)):
            x = self.bns[i-1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout , training=self.training)
            x = self.convs[i](x, edge_index)
            if i%2 == 0:
                x = x + (self.skip_coefficient * x_skip)
                x_skip = x
        x = self.convs[-1](x, edge_index)
        return x
"""
GAT Skip 2 layer linear_in_out add bns,relu,dropout after linear
"""
class GAT_skip_2_layer_linear_in_out_3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels*in_head))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(torch.nn.Linear(hidden_channels*in_head, out_channels))
        
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
GAT Skip 2 layer linear_in_out default, every layer
"""
class GAT_skip_2_layer_linear_in_out_4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels*in_head))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(torch.nn.Linear(hidden_channels*in_head, out_channels))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        #x = F.dropout(x, p=self.dropout, training=self.training)
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
GAT Skip 2 layer linear_in_out default, every layer, plus bns, relu, dropout
"""
class GAT_skip_2_layer_linear_in_out_5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, skip_coefficient, in_head, out_head):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels*in_head))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        for _ in range(num_layers -2):
            self.convs.append(GATConv(hidden_channels*in_head, hidden_channels, in_head, dropout=self.dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*in_head))
        self.convs.append(torch.nn.Linear(hidden_channels*in_head, out_channels))
        
        self.skip_coefficient = skip_coefficient

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = []
        x = self.convs[0](x)
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
            wandb.summary["gat/highest-train-mean"] = r.mean()
            wandb.summary["gat/highest-train-std"] = r.std()
            wandb.log({"gat/highest-train-mean": r.mean(), "gat/highest-train-std": r.std()})
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gat/highest-valid-mean"] = r.mean()
            wandb.summary["gat/highest-valid-std"] = r.std()
            wandb.log({"gat/highest-valid-mean": r.mean(), "gat/highest-valid-std": r.std()})
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gat/final-train-mean"] = r.mean()
            wandb.summary["gat/final-train-std"] = r.std()
            wandb.log({"gat/final-train-mean": r.mean(), "gat/final-train-std": r.std()})
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            wandb.summary["gat/final-test-mean"] = r.mean()
            wandb.summary["gat/final-test-std"] = r.std()
            wandb.log({"gat/final-test-mean": r.mean(), "gat/final-test-std": r.std()})
            r = best_result[:, 4]
            print(f' Highest Test: {r.max():.2f} ± {r.std():.2f}')
            wandb.summary["gat/highest-test-max"] = r.max()
            wandb.summary["gat/highest-test-std"] = r.std()
            wandb.log({"gat/highest-test-max": r.max(), "gat/highest-test-std": r.std()})

"""
Train/Test
"""
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
    
    for run in range(2):
        if enable_wandb:
            #wandb.init()
            wandb.watch(model)
            summary = dict()
            wandb.summary = summary
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, wandb.config.epochs+1):
            loss = train(model, data, optimizer, criterion)
            wandb.log({"gat/loss": loss})
            result = test(model, data, evaluator)
            logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()
    
    if enable_wandb:
        wandb.finish()

device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)

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
    i = wandb.config.model
    if i == 0:
        model = GAT_Default(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 1:
        model = GAT_Multilayer(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 2:
        model = GAT_skip_1_layer_default(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 3:
        model = GAT_skip_2_layer_default(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 4:
        model = GAT_skip_2_layer_pre_activation(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 5:
        model = GAT_skip_2_layer_linear_in_out_3(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    if i == 6:
        model = GAT_skip_2_layer_sum_all(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes, wandb.config.dropout, wandb.config.num_layers, wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    logger = Logger(2)
    runOGB(data, model, logger)
    print(model)
    torch.cuda.empty_cache()

wandb.agent(sweep_id_arxiv, project="GAT_testing_Arxiv", function=runArxivDate)
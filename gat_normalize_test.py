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
from torch_geometric.nn import GATConv

"""
Download datasets
"""

# transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])

ogbndataset = PygNodePropPredDataset(name='ogbn-arxiv', root='dataset/', transform=T.ToSparseTensor())
dataArxiv = ogbndataset[0]  # Get the first graph object.

print()
print(f'Dataset: {ogbndataset}:')
print('======================')

print()
print(dataArxiv)
print('===========================================================================================================')

split_idx = ogbndataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]


def normalizeFeatures(data):
    for i in range(data.x.size(0)):
        data.x[i] = data.x[i] - data.x[i].min()
        data.x[i].div_(data.x[i].max())
    return data


def normalizeFeatuesMeanStd(data):
    for i in range(data.x.size(0)):
        data.x[i] = (data.x[i]-data.x[i].mean()).div_(data.x[i].std())
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
Add date of publication as normalized value mean and std, min: [old<new], max: [old>new]
"""
def dateNormalizedMeanStd(data, minormax):
    data.node_year = data.node_year.squeeze(1).float()
    if minormax == "min":
        data.node_year = (data.node_year-data.node_year.mean()).div_(data.node_year.std())
    elif minormax == "max":
        data.node_year = (data.node_year*-1)
        data.node_year = (data.node_year - data.node_year.mean()).div_(data.node_year.std())
    else:
        print("Selected min or max")
        return
    data.node_year = data.node_year.unsqueeze(1)
    a = torch.empty(169343, 129)
    for i in range(len(data.x)):
        a[i] = torch.cat([data.x[i], data.node_year[i]], dim=0)
    data.x = a
    return data


dataArxiv.adj_t = dataArxiv.adj_t.to_symmetric()

dataNormalizeSumToOne = dataArxiv.clone().detach()
dataNormalizeBetweenOneZero = dataArxiv.clone().detach()
dataNormalizeMeanStd = dataArxiv.clone().detach()

print(dataNormalizeBetweenOneZero.x.size())
dataNormalizeBetweenOneZero.x = torch.transpose(dataNormalizeBetweenOneZero.x,0,1)
print(dataNormalizeBetweenOneZero.x.size())
dataNormalizeBetweenOneZero = normalizeFeatures(dataNormalizeBetweenOneZero)
dataNormalizeBetweenOneZero.x = torch.transpose(dataNormalizeBetweenOneZero.x,0,1)
print("dataNormalizeBetweenOneZero")
print(dataNormalizeBetweenOneZero.x)
print('===========================================================================================================')

print(dataNormalizeBetweenOneZero.x.size())
dataNormalizeMeanStd.x = torch.transpose(dataNormalizeMeanStd.x,0,1)
print(dataNormalizeMeanStd.x.size())
dataNormalizeMeanStd = normalizeFeatuesMeanStd(dataNormalizeMeanStd)
dataNormalizeMeanStd.x = torch.transpose(dataNormalizeMeanStd.x,0,1)
print("dataNormalizeMeanStd")
print(dataNormalizeMeanStd.x)
print('===========================================================================================================')


dataArxivDateNormalizedMeanStdMin = dataNormalizeMeanStd.clone().detach()
dataArxivDateNormalizedMeanStdMin = dateNormalizedMeanStd(dataArxivDateNormalizedMeanStdMin, "min")
print()
print(dataArxivDateNormalizedMeanStdMin)
print('===========================================================================================================')

dataArxivDateNormalizedMeanStdMax = dataNormalizeMeanStd.clone().detach()
dataArxivDateNormalizedMeanStdMax = dateNormalizedMeanStd(dataArxivDateNormalizedMeanStdMax, "max")
print()
print(dataArxivDateNormalizedMeanStdMax)
print('===========================================================================================================')


dataArxivDateNormalizedZeroOne = dataNormalizeBetweenOneZero.clone().detach()
dataArxivDateNormalizedZeroOne = dateNormalizedZeroOne(dataArxivDateNormalizedZeroOne, "min")
print()
print(dataArxivDateNormalizedZeroOne)
print('===========================================================================================================')

dataArxivDateNormalizedOneZero = dataNormalizeBetweenOneZero.clone().detach()
dataArxivDateNormalizedOneZero = dateNormalizedZeroOne(dataArxivDateNormalizedOneZero, "max")
print()
print(dataArxivDateNormalizedOneZero)
print('===========================================================================================================')


dataArxivDateNormalizedMeanStdMin9 = dataArxiv.clone().detach()
dataArxivDateNormalizedMeanStdMin9 = dateNormalizedMeanStd(dataArxivDateNormalizedMeanStdMin9, "min")
print()
print(dataArxivDateNormalizedMeanStdMin9)
print('===========================================================================================================')

dataArxivDateNormalizedMeanStdMax8 = dataArxiv.clone().detach()
dataArxivDateNormalizedMeanStdMax8 = dateNormalizedMeanStd(dataArxivDateNormalizedMeanStdMax8, "max")
print()
print(dataArxivDateNormalizedMeanStdMax8)
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
    "name": "gat-test-normalize-10-runs",
    "method": "grid",
    "parameters": {
        "hidden_channels": {
            "value": 128
        },
        "weight_decay": {
            "value": 0
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
            "value": 0.5
        },
        "num_layers": {
            "value": 4
        },
        "model": {
            "value": 6
        },
        "in_head": {
            "value": 1
        },
        "out_head": {
            "value": 1
        },
        "date_id": {
            "value": 0
        },
        "normalize_id": {
            "values": [0,1,2]
        }
    }
}

#Register the Sweep with W&B

#sweep_id_cora = wandb.sweep(sweep_config, project="GAT_testing_Cora")
#sweep_id_citeseer = wandb.sweep(sweep_config, project="GAT_testing_CiteSeer")
#sweep_id_pubmed = wandb.sweep(sweep_config, project="GAT_testing_Pubmed")
sweep_id_arxiv = wandb.sweep(sweep_config, project="GAT_testing_Arxiv")

"""
GAT Skip 2 layer linear_in_out add bns,relu,dropout after linear
"""
class GAT_skip_2_layer_linear_in_out_8(torch.nn.Module):
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
        x_start = x
        for i in range(1,(len(self.convs)-1)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i%2 == 0:
                x = x + x_start + (self.skip_coefficient * out[i-2])
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

        for epoch in range(1, wandb.config.epochs + 1):
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
Run all normalize ids
"""


def runArxivNormalize():
    wandb.init()
    k = wandb.config.normalize_id
    i = wandb.config.date_id
    if k == 0:
        print("Start dataArxiv")
        data = dataArxiv
        if i == 8:
            print("Start dataArxivDateNormalizedMeanStdMax8")
            data = dataArxivDateNormalizedMeanStdMax8
        if i == 9:
            print("Start dataArxivDateNormalizedMeanStdMin")
            data = dataArxivDateNormalizedMeanStdMin9
    if k == 1:
        print("Start dataNormalizeBetweenOneZero")
        data = dataNormalizeBetweenOneZero
        if i == 6:
            print("Start dataArxivDateNormalizedOneZero")
            data = dataArxivDateNormalizedOneZero
        if i == 7:
            print("Start dataArxivDateNormalizedZeroOne")
            data = dataArxivDateNormalizedZeroOne
    if k == 2:
        print("Start dataNormalizeMeanStd")
        data = dataNormalizeMeanStd
        if i == 8:
            print("Start dataArxivDateNormalizedMeanStdMax")
            data = dataArxivDateNormalizedMeanStdMax
        if i == 9:
            print("Start dataArxivDateNormalizedMeanStdMin")
            data = dataArxivDateNormalizedMeanStdMin
    print()
    print(data)
    print('===========================================================================================================')
    data = data.to(device)
    print("Model 6")
    model = GAT_skip_2_layer_linear_in_out_8(data.num_features, wandb.config.hidden_channels, ogbndataset.num_classes,
                                             wandb.config.dropout, wandb.config.num_layers,
                                             wandb.config.skip_coefficient, wandb.config.in_head, wandb.config.out_head).to(device)
    logger = Logger(10)
    print(sum(p.numel() for p in model.parameters()))
    runOGB(data, model, logger)
    print(model)
    torch.cuda.empty_cache()


# wandb.agent(sweep_id_test, project="uncategorized", function=runArxivNormalize)
wandb.agent(sweep_id_arxiv, project="GAT_testing_Arxiv", function=runArxivNormalize)

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

planetoiddatasetCora = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())

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

planatoiddatasetCiteSeer = Planetoid(root='data/Planatoid', name='CiteSeer', transform=T.NormalizeFeatures())

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

planatoiddatasetPubMed = Planetoid(root='data/Planatoid', name='PubMed', transform=T.NormalizeFeatures())

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

ogbndataset = PygNodePropPredDataset(name='ogbn-arxiv', root = 'dataset/', transform=T.ToSparseTensor())
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

    for run in range(2):
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
        out = model(data.x, data.edge_index, data.edge_weight)[train_idx]
        loss = criterion(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        return loss

    def test(model, data, evaluator):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
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
    logger = Logger(2)
    runPlanetoid(dataCora, model, logger)
    print(model)
    print(dataCora)

#wandb.agent(sweep_id_test, project="uncategorized", function=runCora)
#wandb.agent(sweep_id_cora, project="GCN_testing_Cora", function=runCora)

def runCiteseer():
    wandb.init()
    global dataCiteSeer
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
    logger = Logger(2)
    runPlanetoid(dataCiteSeer, model, logger)
    print(model)
    print(dataCiteSeer)

#wandb.agent(sweep_id_citeseer, project="GCN_testing_CiteSeer", function=runCiteseer)

def runPubmed():
    wandb.init()
    global dataPubMed
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
    logger = Logger(2)
    runPlanetoid(dataPubMed, model, logger)
    print(model)
    print(dataPubMed)

#wandb.agent(sweep_id_pubmed, project="GCN_testing_Pubmed", function=runPubmed)
    
dataArxiv.adj_t = dataArxiv.adj_t.to_symmetric()
dataArxiv = dataArxiv.to(device)

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
    logger = Logger(2)
    runOGB(dataArxivWeight, model, logger)
    print(model)

#wandb.agent(sweep_id_test, project="uncategorized", function=runArxiv)
#wandb.agent(sweep_id_arxiv, project="GCN_testing_Arxiv", function=runArxiv)


import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.nn import GCNConv, GATConv

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
GAT Default
"""

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
        x = F.dropout(x, p=self.dropout, training=self.training)
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
        x = F.dropout(x, p=self.dropout, training=self.training)
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
        

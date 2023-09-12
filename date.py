import torch

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

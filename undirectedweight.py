import torch

"""
Preform to_undirected and add edge_weight to graph, old edges are 2, new edges are 1
"""
def undirectedWeight(dataArxivWeight):
  dataArxivWeight.edge_weight = torch.ones((dataArxivWeight.edge_index.size(1), ), dtype=dataArxivWeight.x.dtype)
  row, col = dataArxiv.edge_index[0], dataArxiv.edge_index[1]
  row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
  edge_index = torch.stack([row, col], dim=0)
  edge_weight = torch.cat([dataArxivWeight.edge_weight * 2, dataArxivWeight.edge_weight], dim=0)
  dataArxivWeight.edge_index, dataArxivWeight.edge_weight = U.coalesce(edge_index,edge_weight)
  return dataArxivWeight
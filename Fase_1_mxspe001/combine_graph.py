import torch

graph = torch.load("graph_10_p2.pt", weights_only=False)
labels = torch.load("labels_10_p2.pt", weights_only=False)

graph.y = labels
torch.save(graph, "gnn_data_10_p2.pt")
print(f"Graph nodes: {graph.x.shape[0]}")
print(f"Features: {graph.x.shape[1]}")
if graph.edge_index is not None and graph.edge_index.numel() > 0:
    print(f"Edges: {graph.edge_index.shape[1]}")
else:
    print("Edges: 0 (graph has no edges)")
print(f"Labels: {graph.y.tolist()}")

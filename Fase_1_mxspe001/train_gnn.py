import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Carregar dados (com weights_only=False)
data = torch.load("gnn_data_10_p2.pt", weights_only=False)

# Verificar se há arestas; se não, adicionamos self-loops para pelo menos conectar cada nó a si mesmo?
if data.edge_index is None or data.edge_index.numel() == 0:
    print("Aviso: grafo sem arestas. Adicionando self-loops para permitir treinamento.")
    data.edge_index = torch.arange(data.num_nodes).view(1, -1).repeat(2, 1)

# Definir máscaras de treino/validação (apenas 2 nós, então usamos um para treino e outro para teste)
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[0] = True   # nó 0 treina
val_mask[1] = True     # nó 1 valida/teste

# Modelo GNN (GraphSAGE)
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Inicializar modelo, otimizador e função de perda
model = GNN(data.num_features, hidden_channels=16, out_channels=4)  # 4 classes (k=0..3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Loop de treinamento
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[val_mask] == data.y[val_mask]).sum().item()
    acc = correct / val_mask.sum().item()
    return acc

for epoch in range(100):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {acc:.2f}")

# Predição final
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)
print("\nResultados finais:")
print(f"Rótulos reais: {data.y.tolist()}")
print(f"Predições: {pred.tolist()}")

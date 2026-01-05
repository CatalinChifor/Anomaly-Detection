import torch
import scipy.io as sio
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mat = sio.loadmat('ACM.mat')
x = torch.tensor(mat['Attributes'].toarray(), dtype=torch.float32).to(device)
adj_sparse = mat['Network']
labels = mat['Label'].flatten()
edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
edge_index = edge_index.to(device)

class Encoder(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, 128)
        self.c2 = GCNConv(128, 64)
    def forward(self, x, ei):
        return F.relu(self.c2(F.relu(self.c1(x, ei)), ei))

class AttrDec(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = GCNConv(64, 128)
        self.c2 = GCNConv(128, in_dim)
    def forward(self, z, ei):
        return self.c2(F.relu(self.c1(z, ei)), ei)

class StructDec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = GCNConv(64, 64)
    def forward(self, z, ei):
        return F.relu(self.c(z, ei))

class GAE(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.ad = AttrDec(in_dim)
        self.sd = StructDec()
    def forward(self, x, ei):
        z = self.enc(x, ei)
        return self.ad(z, ei), self.sd(z, ei)

model = GAE(x.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.004)

for e in range(1, 101):
    model.train()
    opt.zero_grad()
    xh, s = model(x, edge_index)
    
    loss_x = torch.norm(x - xh, p='fro')**2
    
    st_s = s.t() @ s
    norm_ss_sq = torch.sum(st_s * st_s) 
    row, col = edge_index
    inner_prod = torch.sum(torch.sum(s[row] * s[col], dim=1))
    loss_a = norm_ss_sq - 2 * inner_prod
    
    loss = 0.8 * loss_x + 0.2 * loss_a
    loss.backward()
    opt.step()

    if e % 5 == 0:
        model.eval()
        with torch.no_grad():
            xh, _ = model(x, edge_index)
            err = torch.norm(x - xh, p=2, dim=1).cpu().numpy()
            print(f"Epoch {e:03d} | AUC: {roc_auc_score(labels, err):.4f}")
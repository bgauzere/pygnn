import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphPooling


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2*nhid)
        self.dropout = dropout
        self.pooling = GraphPooling()
        self.fc3 = nn.Linear(2*nhid, nhid)
        self.fc4 = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.pooling(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=0)
        return x

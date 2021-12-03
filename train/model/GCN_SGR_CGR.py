import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class get_softadj(nn.Module):
    def __init__(self, features):
        super(get_softadj, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(features, features))
        self.W2 = nn.Parameter(torch.FloatTensor(features, features))

        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)

    def forward(self, x):
        g = torch.matmul(torch.matmul(self.W1, x).permute(0, 2, 1), torch.matmul(self.W2, x))
        g = F.softmax(g, dim=2)
        return g

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x

class GraphConv(nn.Module):
    def __init__(self, num_state, agg):
        super(GraphConv, self).__init__()
        self.num_state = num_state
        self.weight = nn.Parameter(
            torch.FloatTensor(num_state, num_state))
        self.agg = agg()
        init.xavier_uniform_(self.weight)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.num_state)
        agg_feats = self.agg(features, A)
        out = torch.einsum('bnd,df->bnf', (agg_feats, self.weight))
        out = F.relu(out)
        return out

class gcn(nn.Module):
    def __init__(self, num_state):
        super(gcn, self).__init__()
        self.conv1 = GraphConv(num_state, MeanAggregator)

    def forward(self, x, A, train=True):
        x = self.conv1(x, A)
        return x

class GCN_SGR(nn.Module):

    def __init__(self, in_channel=512, state_channel=512, node_num=256, normalize=True):

        super(GCN_SGR, self).__init__()

        self.normalize = normalize

        self.state = state_channel      # the channel of the node feature

        self.node_num = node_num        # the number of the node feature

        self.conv_head = nn.Conv2d(in_channel, self.state, kernel_size=1)

        self.conv_proj = nn.Conv2d(in_channel, self.state, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(int(self.node_num ** 0.5))

        self.conv_adj = get_softadj(self.state)

        self.gcn = gcn(self.state)

        self.conv_tail = nn.Conv2d(self.state, in_channel, kernel_size=1, bias=False)

        self.blocker = nn.BatchNorm2d(in_channel, eps=1e-04)

    def forward(self, x):

        # x: B C H W
        Batch = x.size(0) # B

        K = self.conv_head(x).view(Batch, self.state, -1)  # B State L, L = H*W

        Dynamic_proj = self.conv_proj(x)  # B S H W

        grid = self.pool(Dynamic_proj).reshape(Batch, self.state, -1)  # B S N

        P = grid.permute(0, 2, 1) # B N S

        T = Dynamic_proj.reshape(Batch, self.state, -1)  # B S L

        B_s = torch.matmul(P, T)  # B N S* B S L  (B N L)

        B_s = F.softmax(B_s, dim=1)  # B N L

        D_s = B_s  # B L N

        V_s = torch.matmul(K, B_s.permute(0, 2, 1))

        if self.normalize:
            V_s = V_s * (1. / K.size(2))

        adj = self.conv_adj(V_s)

        V_s_rel = self.gcn(V_s.permute(0, 2, 1), adj)

        Y_s_reshaped = torch.matmul(V_s_rel.permute(0, 2, 1), D_s)

        Y_s = Y_s_reshaped.view(Batch, self.state, *x.size()[2:])

        out = x + self.blocker(self.conv_tail(Y_s))

        return out


class GCN_CGR(nn.Module):

    def __init__(self, in_channel=512, state_channel=256, node_num=512):
        super(GCN_CGR, self).__init__()

        self.in_channels = in_channel  # the channel of the input feature

        self.state = state_channel  # the channel of the node feature

        self.node_num = node_num  # the number of the node feature

        self.head = nn.Conv2d(self.in_channels, node_num, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(int(self.state ** 0.5))

        self.conv_adj = get_softadj(self.state)

        self.gcn = gcn(self.state)

        self.reproj = nn.Conv2d(self.in_channels, self.state, kernel_size=1) 

        self.tail = nn.Conv2d(node_num, self.in_channels, kernel_size=1, bias=False)

        self.blocker = nn.BatchNorm2d(self.in_channels, eps=1e-04)

    def forward(self, x):

        Batch, in_channels, H, W = x.shape  # B C H W

        V_c = (self.pool(self.head(x))).view(Batch, self.node_num, self.state) # B N S

        adj = self.conv_adj(V_c.permute(0, 2, 1))  # B N N

        V_c_rel = self.gcn(V_c, adj)  # B N S

        D_c = self.reproj(x).view(Batch, self.state, H * W)  # B S L

        Y_c_reshaped = torch.matmul(V_c_rel, D_c)  # B N L

        Y_c = Y_c_reshaped.view(Batch, self.node_num, H, W)  # B N H W

        x_c = self.tail(Y_c)  # B C H W

        out = x + self.blocker(x_c)  # B C H W

        return out










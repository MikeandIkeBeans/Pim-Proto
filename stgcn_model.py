import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, K=3, stride=1, dilation=1, dropout=0.2):
        super().__init__()
        self.K = A.size(0)  # partitions
        self.register_buffer('A', A)  # [K,V,V]

        # Spatial (graph) conv: produce K partitions then apply A
        self.gcn = nn.Conv2d(in_c, out_c * self.K, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

        # Temporal conv over T (dims: N,C,T,V → treat V as channels axis via 2D conv with kernel (k,1))
        k = 9
        pad = ((k - 1) // 2) * dilation
        self.tcn = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(k,1),
                      padding=(pad,0), dilation=(dilation,1), bias=False, stride=(stride,1)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout)
        )

        # Residual
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride,1), bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        N,C,T,V = x.shape
        y = self.gcn(x)                          # [N, out_c*K, T, V]
        y = y.view(N, -1, self.K, T, V)          # split partitions
        # sum over partitions after applying A_k
        y = torch.stack([torch.einsum('nctv,vw->nctw', y[:,:,k], self.A[k]) for k in range(self.K)], dim=0).sum(0)
        y = self.bn1(y)
        y = self.act(y)
        y = self.tcn(y)
        res = x if self.down is None else self.down(x)
        return self.act(y + res)

class STGCN(nn.Module):
    def __init__(self, num_classes, A, in_c=3):
        super().__init__()
        chans = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        dil =   [1,   1,  2,   1,   2,   3,   1,   2,   3]
        blocks = []
        c_prev = in_c
        for i,(c,d) in enumerate(zip(chans, dil)):
            stride = 2 if i in {3,6} else 1  # temporal downsample a bit
            blocks.append(STGCNBlock(c_prev, c, A, K=A.size(0), stride=stride, dilation=d))
            c_prev = c
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # global over T,V
        self.fc = nn.Linear(c_prev, num_classes)

    def forward(self, x):  # x: [N,C,T,V]
        z = self.backbone(x)
        z = self.pool(z).squeeze(-1).squeeze(-1) # [N,C]
        return self.fc(z)

class STGCNTwoStream(nn.Module):
    def __init__(self, num_classes, A, in_c=3):
        super().__init__()
        self.joints = STGCN(num_classes, A, in_c)
        self.bones  = STGCN(num_classes, A, in_c)
        # learnable fusion gate
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, xj, xb):  # each: [N,C,T,V]
        lj = self.joints(xj)
        lb = self.bones(xb)
        w = torch.sigmoid(self.alpha)
        return w*lj + (1-w)*lb
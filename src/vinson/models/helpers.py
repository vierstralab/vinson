import torch 

class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)

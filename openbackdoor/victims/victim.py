import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch.nn import init
import math
from typing import *
from opendelta.basemodel import DeltaBase

class Victim(nn.Module):
    def __init__(self):
        super(Victim, self).__init__()

    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass

class MultiScaleLowRankLinear(nn.Module):
    __constants__ = ['in_features', 'out_features', 'inner_rank', 'alpha', 'freqBand']
    in_features: int
    out_features: int
    inner_rank: int
    freqBand: List[int]
    alpha : float
    L: torch.FloatTensor
    R: torch.FloatTensor
    bias: torch.FloatTensor
    weight: torch.FloatTensor
    K: torch.FloatTensor # frequency band width
    def __init__(
        self, 
        in_features: int,  out_features: int, 
        freqBand:List[Union[float, int]], 
        shortcut:bool = True,
        oriLinear:Optional[nn.Linear]=None,
        bias: bool = False, inner_rank:int = 2,
        dropout: float = 0.0, alpha: float = 16.0
    ):
        super(MultiScaleLowRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inner_rank = inner_rank
        self.freqBand = freqBand
        self.bandWidth = len(freqBand)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else lambda x:x
        self.alpha = alpha
        self.scaling = alpha / self.inner_rank
        self.shortcut = shortcut
        
        
        
        if not self.shortcut and oriLinear is not None:
            # retain original Linear Module and set it untunable
            self.oriLinear = nn.Linear(in_features=in_features, out_features=out_features, bias=oriLinear.bias is not None)
            self.oriLinear.weight.data.copy_(oriLinear.weight.data)
            self.oriLinear.weight.requires_grad = False
            if oriLinear.bias is not None:
                self.oriLinear.bias.data.copy_(oriLinear.bias.data)
                self.oriLinear.bias.requires_grad = False
        
        
        self.L = Parameter(torch.empty(self.bandWidth * self.inner_rank, in_features)) # [BW * IR, I]
        self.register_buffer("K", torch.Tensor([[i] * self.inner_rank for i in self.freqBand]).reshape(-1)) # [BW * IR]
        
        self.R = Parameter(torch.empty(out_features, self.bandWidth * self.inner_rank)) # [O, BW * IR]
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def to(self, device):
        self.L = self.L.to(device)
        self.R = self.R.to(device)
        self.K = self.K.to(device)
        self.weight = self.weight.to(device)
        
        if self.bias is not None:
            self.bias = self.bias.to(device)
        if not self.shortcut and hasattr(self, 'oriLinear'):
            self.oriLinear = self.oriLinear.to(device)
        
        return self
            
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.L, a=math.sqrt(5))
        # init.zeros_(self.R)
        init.kaiming_uniform_(self.R, a=math.sqrt(5))
        # init.zeros_(self.L)
        # init.zeros_(self.R)
        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.L)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # init.uniform_(self.bias, -bound, bound)
            init.zeros_(self.bias)
        
              
    def forward(self, X:torch.FloatTensor):
        """
        X: [B, L, I]
        output = X @ W^T + oriBias + X @ (R @ (L \odot K))^T + bias
        """
        multiScaleLowRankOutput = X.matmul((self.R.matmul(self.L * self.K.unsqueeze(0).t())).t())
        if self.bias is not None: 
            multiScaleLowRankOutput += self.bias 
        multiScaleLowRankOutput = multiScaleLowRankOutput * self.scaling
        if not self.shortcut and hasattr(self, 'oriLinear'):
            multiScaleLowRankOutput += self.oriLinear.forward(X)
        return multiScaleLowRankOutput
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, inner_rank={self.inner_rank}, alpha={self.alpha}, freqBand={self.freqBand}, bias={self.bias is not None}'          

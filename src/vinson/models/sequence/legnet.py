from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, filter_sizes: Union[int, list], groups: int = 1):
        super(StemConv, self).__init__()

        filter_sizes = [filter_sizes] if isinstance(filter_sizes, int) else filter_sizes
        num_blocks = len(filter_sizes)
        assert out_ch % num_blocks == 0 

        groups = groups//num_blocks if groups != 1 else 1
        self.convs_list = nn.ModuleList()
        for size in filter_sizes:
            conv = nn.Conv1d(
                        in_channels= in_ch,
                        out_channels = out_ch//num_blocks, 
                        kernel_size= size, 
                        padding='same',
                        bias=False,
                        groups=groups)
            self.convs_list.append(conv)
    
    def forward(self, x):
        out = []
        for conv in self.convs_list:
            out.append(conv(x))
        out = torch.cat(out, dim = -2)
        return out


class SELayer(nn.Module):
    def __init__(self, inp: int, reduction: int = 4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(inp, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), inp),
                nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 ks: int, 
                 resize_factor:int, 
                 out_ch: int = None, 
                 se_reduction: int = None, 
                 activation=nn.SiLU
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor

        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim,
                            padding='same',
                            bias=False
                       ),

                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       SELayer(self.inner_dim, reduction=self.se_reduction),
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
        )
        
        self.block = block

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    def __init__(self, in_ch: int, ks: int, out_ch: int = None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                kernel_size=self.ks,
                padding='same',
                bias=False
            )
        )        
        
    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features,
                      out_channels=out_features, 
                      kernel_size=1),
        )
        
    def forward(self, x):
        return self.block(x) 


class LegNetTrunk(nn.Module):
    def __init__(self, 
                 in_ch: int,
                 stem_ch: int,
                 stem_ks : int, 
                 ef_ks: int,
                 ef_block_sizes: Union[int, list],
                 pool_sizes: list,
                 resize_factor: int,
                 activation=nn.SiLU) -> None:
        
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        
        self.stem_ch = stem_ch
        self.ef_block_sizes = ef_block_sizes
        self.stem = StemConv(
            in_ch=in_ch,
            out_ch=stem_ch,
            filter_sizes=stem_ks
        )
        
        blocks = []
        in_ch = self.stem_ch
        out_ch = self.stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                nn.BatchNorm1d(in_ch), 
                activation(),
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch, 
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation
                    )
                ),
                LocalBlock(
                    in_ch=in_ch * 2,
                    out_ch=out_ch,
                    ks=ef_ks
                ),
                nn.MaxPool1d(pool_sz,) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)
        
        block_ids = [f'blc{blc_id}' for blc_id in range(len(self.ef_block_sizes))]
        blocks_dict = dict(zip(block_ids,blocks))
        self.blocks_dict = nn.ModuleDict(blocks_dict)
        
        self.mapper = MapperBlock(
            in_features=out_ch, 
            out_features=out_ch * 2
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blc_id in range(len(self.ef_block_sizes)):
            cur_block = f'blc{blc_id}'
            x = self.blocks_dict[cur_block](x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        # head is added in lightning wrapper
        return x 


class LegNetTrunkEmbed(LegNetTrunk):
    def __init__(
            self, 
            in_ch: int,
            stem_ch: int,
            stem_ks : int, 
            ef_ks: int,
            ef_block_sizes: Union[int, list],
            pool_sizes: list,
            resize_factor: int,
            n_embed_outputs: int,
            activation=nn.SiLU,
        ) -> None:
        super().__init__(
            in_ch=in_ch,
            stem_ch=stem_ch,
            stem_ks=stem_ks,
            ef_ks=ef_ks,
            ef_block_sizes=ef_block_sizes,
            pool_sizes=pool_sizes,
            resize_factor=resize_factor,
            activation=activation,
        )
        
        self.n_embed_outputs = n_embed_outputs
        self.bias = nn.ModuleDict()

        in_ch_list = [self.stem_ch, *self.ef_block_sizes[:-1]] 
        for blc_id, in_ch in enumerate(in_ch_list):
            cur_block = f'blc{blc_id}'
            self.bias[cur_block] = nn.Linear(n_embed_outputs, in_ch)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        current_layer = self.stem(x)
        
        for blc_id in range(len(self.ef_block_sizes)):
            cur_block = f'blc{blc_id}'
            x_bias = self.bias[cur_block](embed)
            print(x_bias.shape, current_layer.shape)
            current_layer = current_layer + x_bias
            current_layer = self.blocks_dict[cur_block](current_layer)

        x = self.mapper(current_layer)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        return x 


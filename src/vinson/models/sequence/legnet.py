import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    DICT_REDUCTION = dict(none=torch.nn.Identity(),
                          mean=torch.mean,
                          sum=torch.sum)
    def __init__(self, reduction = 'none', pseudocount=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.reduction = GaussianNLLLoss.DICT_REDUCTION[reduction]
        self.pseudocount = pseudocount

    def forward(self, mu1, mu2, target_diff):

        predicted_mean = mu1 - mu2
        predicted_var = mu1 + mu2 + self.pseudocount
        
        loss = 0.5 * (torch.log(predicted_var) + (target_diff - predicted_mean)**2 / predicted_var)
        
        return self.reduction(loss)
        
    
class StemConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, 
                 filter_sizes:list = [6,9,12,15],groups = 1):
        super(StemConv, self).__init__()

        num_blocks = len(filter_sizes)
        assert out_channels % num_blocks == 0, "out_channels should be divisible by the length of filter_sizes"

        groups = groups//num_blocks if groups != 1 else 1
        self.convs_list = []
        for size in filter_sizes:
            conv = nn.Conv1d(in_channels= in_channels, out_channels = out_channels//num_blocks, 
                                            kernel_size= size, padding='same',bias=False,
                                            groups=groups)
            
            self.convs_list.append(conv)
        self.convs_list = nn.ModuleList(self.convs_list)
    
    def forward(self, x):
        lst_out = []
        
        for conv in self.convs_list:
            lst_out.append(conv(x))
        out = torch.cat(lst_out, dim = -2)
        return out


class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
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
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
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
    def __init__(self, in_ch, ks, out_ch=None):
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
                            bias=False)
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
    def __init__(self, in_features, out_features, activation=nn.SiLU):
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
                 in_ch,
                 stem_ch,
                 stem_ks, 
                 ef_ks,
                 ef_block_sizes,
                 pool_sizes,
                 resize_factor,
                 activation=nn.SiLU,
                 out_size = 1,
                 ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        
        self.ef_block_sizes = ef_block_sizes
        self.stem = StemConv(
            in_channels=in_ch,
            out_channels=stem_ch,
            filter_sizes=stem_ks
        )
        
        blocks = []
        sample_mappers = []
        in_ch = stem_ch
        out_ch = stem_ch
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
                        activation=activation)
                ),
                LocalBlock(in_ch=in_ch * 2,
                           out_ch=out_ch,
                           ks=ef_ks),
                nn.MaxPool1d(pool_sz,) if pool_sz != 1 else nn.Identity()
            )
            
            # embed_mapper = nn.Sequential(nn.Linear(637,in_ch),
            #                              nn.BatchNorm1d(in_ch),
            #                              activation(),
            #                              nn.Linear(in_ch, in_ch))
            in_ch = out_ch
            blocks.append(blc)
            # sample_mappers.append(embed_mapper)
        
        block_ids = [f'blc{blc_id}' for blc_id in range(len(self.ef_block_sizes))]
        blocks_dict = dict(zip(block_ids,blocks))
        # sample_mappers_dict = dict(zip(block_ids,sample_mappers))
        self.main = nn.ModuleDict(blocks_dict)
        # self.sample_mappers = nn.ModuleDict(sample_mappers_dict)
        
        self.mapper = MapperBlock(in_features=out_ch, 
                                  out_features=out_ch * 2)
            
    def forward(self, x, sample_repr):
        x = self.stem(x)

        for blc_id in range(len(self.ef_block_sizes)):
            # block, sample_mapper = self.main[f'blc{blc_id}'], self.sample_mappers[f'blc{blc_id}']
            # embed = sample_mapper(sample_repr)
            embed = embed[:,:, None]
            x = x + embed
            # x = block(x)

        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        # head is added in lightning wrapper
        return x 


# FIXME: add bias to LegNetTrunk from embeddings
class LegNetTrunkEmbed(LegNetTrunk):
    def __init__(self, n_embed_outputs: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_embed_outputs = n_embed_outputs

        self.bias2 = torch.nn.Linear(n_embed_outputs, self.layer2[0].out_channels)
        self.bias3 = torch.nn.Linear(n_embed_outputs, self.layer3[0].out_channels)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LegNetTrunkEmbed is not implemented yet.")

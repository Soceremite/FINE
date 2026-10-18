import torch.nn as nn
import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch.nn import MSELoss


# def act_fun(activation='gelu'):
#     if activation=='gelu':
#         return nn.GELU()
#     else:
#         return nn.ReLU()
def act_fun(activation='gelu'):
    func_map = {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }
    return func_map[activation]()  

class TokenAttn(nn.Module):
    def __init__(self, hidden_size, topk_ratio=0.75):
        super(TokenAttn, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        self.topk_ratio=topk_ratio
        
    def forward(self, inputs):
        B,N,D = inputs.shape
        attention = self.attention(inputs).transpose(1,2)
        attention = nn.Softmax(-1)(attention)
        output = attention @ inputs
        pool_output = output.squeeze(1)

        weight = attention.squeeze()
        topk = int(self.topk_ratio * N)
        # 选择TopK特征
        _, topk_indices = weight.topk(topk, dim=-1)
        robust_features = inputs[torch.arange(B).unsqueeze(1), topk_indices]
        
        return robust_features, pool_output


class Multi_Decoder(nn.Module):
    def __init__(self,
                 hidden_size, 
                 nhead=2,
                 ffn_dim=1024,
                 dropout=0.2,
                 activation='relu',
                 num_layers=2,
                 initializer_range=0.02):
        super(Multi_Decoder, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.query.data.normal_(mean=0.0, std=initializer_range)
        decoder_layer = TransformerDecoderLayer(hidden_size, nhead, ffn_dim, dropout,activation,batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size))
        
    def forward(self, memory):
        bs = memory.shape[0]
        query = self.query.expand(bs,-1, -1)
        cls_token = self.decoder(query, memory).squeeze(1)

        return cls_token


class Modal_Decoder(nn.Module):
    def __init__(self,
                 output_dim, 
                 hidden_size, 
                 nhead=2,
                 ffn_dim=1024,
                 dropout=0.2,
                 activation='relu',
                 num_layers=2,
                 initializer_range=0.02,
                 train_mode='regression'):
        super(Modal_Decoder, self).__init__()
        self.train_mode = train_mode
        if self.train_mode == 'classification':
            self.softmax= nn.Softmax(dim=1)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.query.data.normal_(mean=0.0, std=initializer_range)
        decoder_layer = TransformerDecoderLayer(hidden_size, nhead, ffn_dim, dropout,activation,batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size))
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        act_fun(activation),
                                        nn.Linear(hidden_size, output_dim))
        
    def forward(self, memory):
        bs = memory.shape[0]
        query = self.query.expand(bs,-1, -1)
        cls_token = self.decoder(query, memory).squeeze(1)
        logit = self.classifier(cls_token)
        if self.train_mode =='regression':
            logit = logit.squeeze(-1)
        else:
            logit = self.softmax(logit)
        return logit, cls_token
    

class Modal_Decoder2(nn.Module):
    def __init__(self,
                 output_dim, 
                 hidden_size, 
                 nhead=2,
                 ffn_dim=1024,
                 dropout=0.2,
                 activation='relu',
                 num_layers=2,
                 initializer_range=0.02,
                 train_mode='regression'):
        super(Modal_Decoder2, self).__init__()
        self.train_mode = train_mode
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.query.data.normal_(mean=0.0, std=initializer_range)
        decoder_layer = TransformerDecoderLayer(hidden_size, nhead, ffn_dim, dropout,activation,batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size))
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        act_fun(activation),
                                        nn.Linear(hidden_size, output_dim))
        
    def forward(self, memory):
        bs = memory.shape[0]
        query = self.query.expand(bs,-1, -1)
        cls_token = self.decoder(query, memory).squeeze(1)
        logit = self.classifier(cls_token)
        if self.train_mode =='regression':
            logit = logit.squeeze(-1)
        return logit, cls_token


class ASEncoder(nn.Module):
    def __init__(self, 
                 hidden_size=512,
                 output_dim=256,
                 dropout=0.2):
        super(ASEncoder, self).__init__()
        self.output_dim = output_dim

        self.proj = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.Tanh(),
            nn.Linear(hidden_size*2, hidden_size)
        ) for _ in range(3)])
        
        self.shared_feature_extractor= nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, self.output_dim)
        )
 
        self.private_feature_extractor = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, self.output_dim)
        ) for _ in range(3)])

        self.recon = nn.ModuleList([nn.Sequential(
            nn.Linear(self.output_dim*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ) for _ in range(3)])

        self.recon_mse = MSELoss()


    def shared_private(self, feat_t, feat_v, feat_a):


        self.t_pri = self.private_feature_extractor[0](feat_t)
        self.v_pri = self.private_feature_extractor[1](feat_v)
        self.a_pri = self.private_feature_extractor[2](feat_a)

        self.t_sha = self.shared_feature_extractor(feat_t)
        self.v_sha = self.shared_feature_extractor(feat_v)
        self.a_sha = self.shared_feature_extractor(feat_a)

    def reconstruct(self, text, video, audio):

        self.t_add = torch.cat((self.t_pri,self.t_sha),dim=2)
        self.v_add = torch.cat((self.v_pri,self.v_sha),dim=2)
        self.a_add = torch.cat((self.a_pri,self.a_sha),dim=2)

        self.t_recon = self.recon[0](self.t_add)
        self.v_recon = self.recon[1](self.v_add)
        self.a_recon = self.recon[2](self.a_add)

        self.recon_loss = self.recon_mse(self.t_recon, text) \
                            + self.recon_mse(self.v_recon, video) \
                            + self.recon_mse(self.a_recon, audio)
        
    
    def forward(self, video, audio, text):

        text = self.proj[0](text)
        video = self.proj[1](video)
        audio = self.proj[2](audio)
        self.shared_private(text, video, audio)
        self.reconstruct(text, video, audio)
        
        return  self.t_sha, self.v_sha, self.a_sha, \
                self.t_pri, self.v_pri, self.a_pri, \
                self.recon_loss

class CMA(nn.Module):
    def __init__(self, embed_dim, nheads=2, dropout=0.2, ffn_dim=1024, batch_first=True, use_ffn=True, activation='relu'):
        super(CMA, self).__init__()
        self.cma = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nheads, batch_first=batch_first)
        
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.use_ffn = use_ffn
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                act_fun(activation),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim)
            )

    def forward(self, q_feat, va_feat):
        cma_out, _ = self.cma(query=q_feat, key=va_feat, value=va_feat)
        cma_out = self.layernorm1(cma_out + q_feat)  # 残差连接后 LayerNorm
        if self.use_ffn:
            ffn_out = self.ffn(cma_out)
            out = self.layernorm2(cma_out + ffn_out)  # 再做 LayerNorm 和 Dropout
        else:
            out = cma_out
        return out
    
class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='gelu'):
        super(BaseClassifier, self).__init__()
        ModuleList = []
        if len(hidden_size) != 1 or hidden_size[0] != input_size:
            for i, h in enumerate(hidden_size):
                if i == 0:
                    ModuleList.append(nn.Linear(input_size, h))
                else:
                    ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(act_fun(activation))

            ModuleList.append(nn.Linear(hidden_size[-1], output_size))
            self.classifier = nn.Sequential(*ModuleList)
        else:
            self.classifier = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.classifier(x)

# class BaseClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, activation='gelu'):
#         super(BaseClassifier, self).__init__()
#         ModuleList = []
#         if len(hidden_size)!=1 or hidden_size[0]!=input_size:
#             for i, h in enumerate(hidden_size):
#                 if i == 0:
#                     ModuleList.append(nn.Linear(input_size, h))
#                     ModuleList.append(act_fun(activation))
#                 else:
#                     ModuleList.append(nn.Linear(hidden_size[i - 1], h))
#                     ModuleList.append(act_fun(activation))

#         self.MLP = nn.Sequential(*ModuleList)
#         self.fc_out = nn.Linear(hidden_size[-1], output_size)

#     def forward(self, x):
#         feat = self.MLP(x)
#         out = self.fc_out(feat)
#         return out, feat

class CLUBVec2Seq(nn.Module):
    """ The CLUB estimator for vector-to-sequence pairs.
    """
    def __init__(
        self,
        seq_dim: int,
        vec_dim: int,
        hidden_size: int,
        kernel_size: int = 5,
        is_sampled_version: bool = False,
    ):
        super().__init__()
        self.is_sampled_version = is_sampled_version
        assert kernel_size in [3,5,7]

        padding = kernel_size//2

        self.seq_prenet = nn.Sequential(
            nn.Conv1d(seq_dim, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
        )
        # mu net
        self.p_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim)
        )
        # variance net
        self.p_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim),
            nn.Tanh()
        )

    def temporal_avg_pool(self, x, mask=None):
        """
        Args:
            x (tensor): shape [B, T, D]
            mask (bool tensor): padding parts with ones
        """
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def get_mu_logvar(self, seq, mask):
        # [B, T, D]
        h = self.seq_prenet(seq.transpose(1, 2)).transpose(1, 2)
        # [B, D]
        h = self.temporal_avg_pool(h, mask)
        mu = self.p_mu(h)
        logvar = self.p_logvar(h)
        return mu, logvar
        
    def loglikeli(self, seq, vec, mask=None):
        """ Compute un-normalized log-likelihood
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """
        # mu/logvar: (bs, vec_dim)
        mu, logvar = self.get_mu_logvar(seq, mask)
        return (-(mu - vec)**2 /logvar.exp() - logvar).sum(dim=1).mean(dim=0) 

    def forward(self, seq, vec, mask=None):
        """ Estimate mutual information CLUB upper bound.
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """

        mu, logvar = self.get_mu_logvar(seq, mask)

        if self.is_sampled_version:
            sample_size = seq.shape[0]
            #random_index = torch.randint(sample_size, (sample_size,)).long()
            random_index = torch.randperm(sample_size).long()

            positive = - (mu - vec)**2 / logvar.exp()
            negative = - (mu - vec[random_index])**2 / logvar.exp()
            upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
            
            mi_upper = upper_bound / 2.
        else:
            # log of conditional probability of positive sample pairs, [B, D]
            positive = - (mu - vec)**2 /2./logvar.exp()
            # [B, 1, D]
            prediction_1 = mu.unsqueeze(1)
            # [1, B, D]
            y_samples_1 = vec.unsqueeze(0)

            # log of conditional probability of negative sample pairs, [B, D]
            negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

            mi_upper = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        # print(mi_upper)
        
        return mi_upper

    def learning_loss(self, seq, vec, mask=None):
        return - self.loglikeli(seq, vec, mask)
"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)




class MISA(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 task_dim,
                 sim_weight=1.0, 
                 diff_weight=0.3, 
                 recon_weight=1.0):
        super(MISA, self).__init__()

        hidden_dim = hidden_dim
        task_dim= task_dim
        self.sim_weight = sim_weight     # 1.0
        self.diff_weight = diff_weight   # 0.3
        self.recon_weight = recon_weight # 1.0
        

        # map into a common space

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_dim[0], out_features=hidden_dim[0]))
        self.project_a.add_module('project_a_activation', nn.ReLU())
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(hidden_dim[0]))

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_dim[1], out_features=hidden_dim[1]))
        self.project_t.add_module('project_t_activation', nn.ReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(hidden_dim[1]))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_dim[2], out_features=hidden_dim[2]))
        self.project_v.add_module('project_v_activation', nn.ReLU())
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(hidden_dim[2]))


        # private encoders
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=hidden_dim[0], out_features=task_dim[0]))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=hidden_dim[1], out_features=task_dim[1]))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=hidden_dim[2], out_features=task_dim[2]))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        # shared encoder
        assert hidden_dim[0]==hidden_dim[1]==hidden_dim[2]
        assert task_dim[0]==task_dim[1]==task_dim[2]
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=hidden_dim[0], out_features=task_dim[0]))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        # reconstruct
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=task_dim[0], out_features=hidden_dim[0]))
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=task_dim[1], out_features=hidden_dim[1]))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=task_dim[2], out_features=hidden_dim[2]))
        

    def shared_private(self, feat_t, feat_v, feat_a):
        # Projecting to same sized space
        self.t_orig = feat_t = self.project_t(feat_t)
        self.v_orig = feat_v = self.project_v(feat_v)
        self.a_orig = feat_a = self.project_a(feat_a)

        # Private-shared components
        self.pri_t = self.private_t(feat_t)
        self.pri_v = self.private_v(feat_v)
        self.pri_a = self.private_a(feat_a)

        self.shared_t = self.shared(feat_t)
        self.shared_v = self.shared(feat_v)
        self.shared_a = self.shared(feat_a)

        
    def reconstruct(self):
        self.add_t = (self.pri_t + self.shared_t)
        self.add_v = (self.pri_v + self.shared_v)
        self.add_a = (self.pri_a + self.shared_a)

        self.t_recon = self.recon_t(self.add_t)
        self.v_recon = self.recon_v(self.add_v)
        self.a_recon = self.recon_a(self.add_a)

    ##########################################################
    ## inter loss calculation
    ##########################################################
    def get_recon_loss(self):
        loss =  MSE()(self.t_recon, self.t_orig)
        loss += MSE()(self.v_recon, self.v_orig)
        loss += MSE()(self.a_recon, self.a_orig)
        loss = loss / 3.0
        return loss

    def get_diff_loss(self):
        shared_t = self.shared_t
        shared_v = self.shared_v
        shared_a = self.shared_a
        pri_t = self.pri_t
        pri_v = self.pri_v
        pri_a = self.pri_a

        # Between private and shared
        loss =  DiffLoss()(pri_t, shared_t)
        loss += DiffLoss()(pri_v, shared_v)
        loss += DiffLoss()(pri_a, shared_a)

        # Across privates
        loss += DiffLoss()(pri_a, pri_t)
        loss += DiffLoss()(pri_a, pri_v)
        loss += DiffLoss()(pri_t, pri_v)
        return loss

    def get_cmd_loss(self):
        # losses between shared states
        loss =  CMD()(self.shared_t, self.shared_v, 5)
        loss += CMD()(self.shared_t, self.shared_a, 5)
        loss += CMD()(self.shared_a, self.shared_v, 5)
        loss = loss/3.0
        return loss
    
    def forward(self, audio, text, video):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            text_feat:  tensor of shape (batch, seqlen2, text_in)
            video_feat: tensor of shape (batch, seqlen3, video_in)
        '''

        # shared-private encoders
        self.shared_private(text, video, audio)

        # For reconstruction
        self.reconstruct()
        
        interloss = self.diff_weight  * self.get_diff_loss() + \
                    self.sim_weight   * self.get_cmd_loss()  + \
                    self.recon_weight * self.get_recon_loss()
        
        return self.shared_a, self.shared_t, self.shared_v, \
                self.pri_a, self.pri_t, self.pri_v, \
                interloss
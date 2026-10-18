"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import MSELoss
import numpy as np
from torch import optim
from itertools import chain

def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)

class InfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers=0, activation='relu'):
          super(InfoNCECritic, self).__init__()
          # output is scalar score
        #   self.x_proj = mlp(A_dim,hidden_dim,A_dim,0,activation)
          
        #   self.y_proj = mlp(B_dim,hidden_dim,B_dim,0,activation)
          
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples, use_proj=True):
        
        # x_samples = self.x_proj(x_samples)
        # y_samples = self.y_proj(y_samples)
        
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim = -1))  

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound

# Concat critic with the CLUBInfoNCE (NCE-CLUB) objective
class CLUBInfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
          super(CLUBInfoNCECritic, self).__init__()
 
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    # CLUB loss   min upper bound
    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1)) 
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1))  

        return T0.mean() - T1.mean()

    # InfoNCE loss max lower bound
    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1)) 

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound / 2.
    
    # def forward(self, x_samples, y_samples): 
    #     mu, logvar = self.get_mu_logvar(x_samples)
        
    #     # log of conditional probability of positive sample pairs
    #     positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
    #     prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
    #     y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

    #     # log of conditional probability of negative sample pairs
    #     negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

    #     return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class FTREncoder(nn.Module):
    def __init__(self, 
                 input_dim=[256,256,256],
                 shauni_dim=[128,128,128],
                 label_dim=1,
                 shauni_tr_dim=[64,64,64],
                 task_dim=[64,64,64,64],
                 mi_hidden_dim=[512,512,512,512],
                 dropout=0.2,
                 optimizer_mode='adamw',
                 weight_decay=5e-4,
                 activation='relu',
                 mi_lr=1e-4):
        super(FTREncoder, self).__init__()

        self.sha_dim = shauni_dim
        self.uni_dim = shauni_dim
        self.hidden_dim  = mi_hidden_dim
        self.task_dim = task_dim
        self.sha_tr_dim  = shauni_tr_dim
        self.utr_tr_dim  = shauni_tr_dim


        # shared information encoder
        self.shared_information_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(input_dim[i], self.sha_dim[i])
                ) for i in range(3)
            ]
        )
        # unique information encoder
        self.unique_information_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(input_dim[i], self.uni_dim[i])
                ) for i in range(3)
            ]
        )
        # reconstructed decoder
        self.recon = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.sha_dim[i] + self.uni_dim[i], self.sha_dim[i] + self.uni_dim[i]),
                    nn.ReLU(),
                    nn.Linear(self.sha_dim[i] + self.uni_dim[i], input_dim[i])
                ) for i in range(3)
            ]
        )

        self.common_task_proj = nn.Sequential(
            nn.Linear(label_dim, self.task_dim[0] // 2),
            nn.ReLU(),
            nn.Linear(self.task_dim[0] // 2, self.task_dim[0])
        )


        self.unique_task_proj =nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(label_dim, self.task_dim[i+1] // 2),
                    nn.ReLU(),
                    nn.Linear(self.task_dim[i+1] // 2, self.task_dim[i+1])
                ) for i in range(3)
            ]
        )

        self.shared_task_relevant_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.sha_dim[i], self.sha_dim[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(self.sha_dim[i], shauni_tr_dim[i])
                ) for i in range(3)
            ]
        )

        self.unique_task_relevant_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.uni_dim[i], self.uni_dim[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(self.uni_dim[i], shauni_tr_dim[i])
                ) for i in range(3)
            ]
        )


        # mutual information estimator between shared information of two modality max
        self.sha_at = InfoNCECritic(self.sha_dim[0], self.sha_dim[1], self.hidden_dim[0], activation=activation)
        self.sha_tv = InfoNCECritic(self.sha_dim[1], self.sha_dim[2], self.hidden_dim[0], activation=activation)
        self.sha_va = InfoNCECritic(self.sha_dim[2], self.sha_dim[0], self.hidden_dim[0], activation=activation)

        # mutual information estimator between unique information of two modality min
        # self.uni_tv = CLUB(self.uni_dim, self.uni_dim, self.hidden_dim[1])
        # self.uni_va = CLUB(self.uni_dim, self.uni_dim, self.hidden_dim[1])
        # self.uni_at = CLUB(self.uni_dim, self.uni_dim, self.hidden_dim[1])
        self.uni_at = CLUBInfoNCECritic(self.uni_dim[0], self.uni_dim[1], self.hidden_dim[1], 0, activation=activation)
        self.uni_tv = CLUBInfoNCECritic(self.uni_dim[1], self.uni_dim[2], self.hidden_dim[1], 0, activation=activation)
        self.uni_va = CLUBInfoNCECritic(self.uni_dim[2], self.uni_dim[0], self.hidden_dim[1], 0, activation=activation)

        # shared but task_irrelevant  information estimator   min
        self.sti_at = CLUBInfoNCECritic(self.sha_tr_dim[0]+self.task_dim[0], self.sha_tr_dim[1]+self.task_dim[0], self.hidden_dim[2], 0, activation=activation) 
        self.sti_tv = CLUBInfoNCECritic(self.sha_tr_dim[1]+self.task_dim[0], self.sha_tr_dim[2]+self.task_dim[0], self.hidden_dim[2], 0, activation=activation) 
        self.sti_va = CLUBInfoNCECritic(self.sha_tr_dim[2]+self.task_dim[0], self.sha_tr_dim[0]+self.task_dim[0], self.hidden_dim[2], 0, activation=activation) 

        # unique and task_relevant information estimator max
        self.utr_ay = InfoNCECritic(self.utr_tr_dim[0], self.task_dim[1], self.hidden_dim[3], activation=activation)
        self.utr_ty = InfoNCECritic(self.utr_tr_dim[1], self.task_dim[2], self.hidden_dim[3], activation=activation)
        self.utr_vy = InfoNCECritic(self.utr_tr_dim[2], self.task_dim[3], self.hidden_dim[3], activation=activation)


        self.recon_criterion = MSELoss()

        self.optimzers = []

        self.uni_optimizer = optim.Adam(chain(self.uni_at.parameters(),
                                        self.uni_tv.parameters(), 
                                        self.uni_va.parameters()),lr=mi_lr)
        
        self.sti_optimizer = optim.Adam(chain(self.sti_tv.parameters(),
                                        self.sti_va.parameters(), 
                                        self.sti_at.parameters()),lr=mi_lr)
        
        self.optimzers.append(self.uni_optimizer)
        self.optimzers.append(self.sti_optimizer)
    
    def get_optimizers(self):
        return self.optimzers
    
    def parameters(self, prefix = '', recurse = True, remove_duplicate = True):
        mi_list = ['uni_', 'sti_']
        for name, param in super().named_parameters(prefix=prefix, 
                                                    recurse=recurse, 
                                                    remove_duplicate=remove_duplicate):
            if not any(name.startswith(m) for m in mi_list):
                yield param

    def su_encoder(self, feat_a, feat_t, feat_v):

        self.feat_a_sha = self.shared_information_encoder[0](feat_a)
        self.feat_t_sha = self.shared_information_encoder[1](feat_t)
        self.feat_v_sha = self.shared_information_encoder[2](feat_v)

        self.feat_a_uni = self.unique_information_encoder[0](feat_a)
        self.feat_t_uni = self.unique_information_encoder[1](feat_t)
        self.feat_v_uni = self.unique_information_encoder[2](feat_v)
    
    def reconstruct(self, a_feat, t_feat, v_feat):

        add_a = torch.cat((self.feat_a_sha, self.feat_a_uni), dim=2)
        add_t = torch.cat((self.feat_t_sha, self.feat_t_uni), dim=2)
        add_v = torch.cat((self.feat_v_sha, self.feat_v_uni), dim=2)

        recon_a = self.recon[0](add_a)
        recon_t = self.recon[1](add_t)
        recon_v = self.recon[2](add_v)

        return  self.recon_criterion(recon_a, a_feat)  \
                + self.recon_criterion(recon_t, t_feat) \
                + self.recon_criterion(recon_v, v_feat)
    
    def sha_mi(self):

        feat_sha_a = torch.mean(self.feat_a_sha, dim=1)
        feat_sha_t = torch.mean(self.feat_t_sha, dim=1)
        feat_sha_v = torch.mean(self.feat_v_sha, dim=1)

        sha_loss  = self.sha_at(feat_sha_a, feat_sha_t)
        sha_loss += self.sha_tv(feat_sha_t, feat_sha_v)
        sha_loss += self.sha_va(feat_sha_v, feat_sha_a)

        return sha_loss

    def uni_mi(self):
        '''
        unique information of single modal mutual information estimator
        '''
        feat_uni_a = torch.mean(self.feat_a_uni, dim=1)
        feat_uni_t = torch.mean(self.feat_t_uni, dim=1)
        feat_uni_v = torch.mean(self.feat_v_uni, dim=1)
        uni_loss  = self.uni_at(feat_uni_a, feat_uni_t)
        uni_loss += self.uni_tv(feat_uni_t, feat_uni_v)
        uni_loss += self.uni_va(feat_uni_v, feat_uni_a)
        return uni_loss

    def uni_backward(self, mm_iter=3):
        feat_uni_a = torch.mean(self.feat_a_uni, dim=1).detach()
        feat_uni_t = torch.mean(self.feat_t_uni, dim=1).detach()
        feat_uni_v = torch.mean(self.feat_v_uni, dim=1).detach()
        for _ in range(mm_iter):
            self.uni_optimizer.zero_grad()
            mi_loss =  self.uni_at.learning_loss(feat_uni_a, feat_uni_t)
            mi_loss += self.uni_tv.learning_loss(feat_uni_t, feat_uni_v)
            mi_loss += self.uni_va.learning_loss(feat_uni_v, feat_uni_a)
            mi_loss /= 3
            mi_loss.backward()
            self.uni_optimizer.step()

    def sti_mi(self):
        '''
        shared but task_irrelevant mutual information estimator
        '''
        task =  self.task[0]
        a_str = torch.mean(self.feat_a_str, dim=1)
        a_str_t = torch.cat([a_str, task], dim=1)
        t_str = torch.mean(self.feat_t_str, dim=1)
        t_str_t = torch.cat([t_str, task], dim=1)
        v_str = torch.mean(self.feat_v_str, dim=1)
        v_str_t = torch.cat([v_str, task], dim=1)
        
        loss  = self.sti_at(a_str_t, t_str_t)
        loss += self.sti_tv(t_str_t, v_str_t)
        loss += self.sti_va(v_str_t, a_str_t)
        return loss

    def sti_backward(self, mm_iter=3):
        '''
        shared but task_irrelevant  loss backward
        '''
        str_a = torch.mean(self.feat_a_str, dim=1).detach()
        str_t = torch.mean(self.feat_t_str, dim=1).detach()
        str_v = torch.mean(self.feat_v_str, dim=1).detach()
        task = self.task[0].detach()
        str_a_t = torch.cat([str_a, task], dim=1)
        str_t_t = torch.cat([str_t, task], dim=1)
        str_v_t = torch.cat([str_v, task], dim=1)
        for _ in range(mm_iter):
            self.sti_optimizer.zero_grad()
            mi_loss =  self.sti_at.learning_loss(str_a_t, str_t_t)
            mi_loss += self.sti_tv.learning_loss(str_t_t, str_v_t)
            mi_loss += self.sti_va.learning_loss(str_v_t, str_a_t)
            mi_loss /= 3
            mi_loss.backward()
            self.sti_optimizer.step()
    
    def str_encoder(self, a_sha, t_sha, v_sha):

        self.feat_a_str = self.shared_task_relevant_encoder[0](a_sha)
        self.feat_t_str = self.shared_task_relevant_encoder[1](t_sha)
        self.feat_v_str = self.shared_task_relevant_encoder[2](v_sha)

    def utr_encoder(self, a_uni, t_uni, v_uni):

        self.feat_a_utr = self.unique_task_relevant_encoder[0](a_uni)
        self.feat_t_utr = self.unique_task_relevant_encoder[1](t_uni)
        self.feat_v_utr = self.unique_task_relevant_encoder[2](v_uni)

    def utr_mi(self):
        utr_loss  = self.utr_ay(torch.mean(self.feat_a_utr, dim=1), self.task[1])
        utr_loss += self.utr_ty(torch.mean(self.feat_t_utr, dim=1), self.task[2])
        utr_loss += self.utr_vy(torch.mean(self.feat_v_utr, dim=1), self.task[3])
        return  utr_loss
    
    # def forward(self, text, video, audio, label=None):
    def forward(self, feat_a, feat_t, feat_v, label=None):
        self.task = None
        if label !=None:
            if len(label.shape)==1:
                label = label.unsqueeze(1)
            # self.task = self.task_proj(label)
            self.task = [self.common_task_proj(label)]
            for i in range(3):
                self.task.append(self.unique_task_proj[i](label))
        
        self.su_encoder(feat_a, feat_t, feat_v)
        
        if self.task !=None:
            # max shared information between two modality
            share_loss = self.sha_mi()
            # min unique information between two modality
            unique_loss = self.uni_mi()
            # reconstruct
            recon_loss = self.reconstruct(feat_a, feat_t, feat_v)
        
        self.str_encoder(self.feat_a_sha, self.feat_t_sha, self.feat_v_sha)
        self.utr_encoder(self.feat_a_uni, self.feat_t_uni, self.feat_v_uni)
        if  self.task != None:

            # min shared but task-irrelevant information
            sha_ti_loss = self.sti_mi()

            # max unqiue and task-relevant information
            uni_tr_loss = self.utr_mi()
            

        if self.task != None:
            return  self.feat_a_str, self.feat_t_str, self.feat_v_str, \
                    self.feat_a_utr, self.feat_t_utr, self.feat_v_utr, \
                    share_loss, unique_loss, recon_loss,\
                    sha_ti_loss, uni_tr_loss
        else:
            return self.feat_a_str, self.feat_t_str, self.feat_v_str, \
                   self.feat_a_utr, self.feat_t_utr, self.feat_v_utr
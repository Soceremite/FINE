import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.utils.functions import SupConLoss,UACCConLoss,ACCConLoss,RnCLoss
class FeatQueue(nn.Module):
    """
    Dynamic Contrastive Queue
    get_queue: get all features and label
    """
    def __init__(self, num_classes, samples_per_class, 
                 min_queue_len, queue_ratio, hidden_size,
                 temperature, train_mode, criterion_loss='supcon',min_label=-3,max_label=3):
        super(FeatQueue, self).__init__()

        self.queue_ratio = queue_ratio
        self.min_queue_len = min_queue_len
        self.min_label = min_label
        self.max_label = max_label
        self.train_mode = train_mode
        self.is_empty=False
        if self.queue_ratio==0.0 and self.min_queue_len==0:
            self.is_empty=True

        sample_per_class=np.array(samples_per_class)*self.queue_ratio
        queue_len =[]
        for item in sample_per_class:
            queue_len.append(max(self.min_queue_len,int(item)))
        self.queue_len  = np.array(queue_len, dtype=np.int64)

        self.queue_patch = num_classes
        self.queue_start = [0 for i in range(num_classes)]
        self.queue_end = [0 for i in range(num_classes)]
        sum=0
        for i in range(num_classes):
            self.queue_start[i]=sum
            sum +=self.queue_len[i].item()
            self.queue_end[i] = sum
        print(self.queue_len)
        print(self.queue_start)
        print(self.queue_end)

        self.register_buffer('queue',torch.randn(self.queue_len.sum().item(), hidden_size))
        self.register_buffer('queue_label',torch.randn(self.queue_len.sum().item(), 1))
        self.register_buffer('queue_cla_label',torch.randn(self.queue_len.sum().item(), 1))
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        
        # 初始化
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_cla_label = self.queue_label.clone()
        self.criterion_loss = criterion_loss.lower()
        if self.criterion_loss =='uacccon':
            print('FeatQueue Loss','UACCConLoss')
            self.con_loss = UACCConLoss(min_label=self.min_label,
                                        max_label=self.max_label,
                                        temperature=temperature)
        elif self.criterion_loss == 'acccon':
            print('FeatQueue Loss','ACCConLoss')
            self.con_loss = ACCConLoss(min_label=self.min_label,
                                       max_label=self.max_label,
                                       temperature=temperature)
        elif self.criterion_loss == 'rnc':
            print('FeatQueue Loss','RnCLoss')
            self.con_loss = RnCLoss()
        else:
            print('FeatQueue Loss','SupConLoss')
            self.con_loss = SupConLoss(temperature=temperature)
    
    def logit2cla(self,input_tensor):
        '''
        regression to classification
        input:[batchsize, 1]
        '''
        if self.train_mode == 'regression':
            if self.queue_patch==5:  # for SIMS dataset
                ms = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
                input_tensor = torch.clip(input_tensor,min=-1., max=1.)
            elif self.queue_patch==7:   # for MOSI MOSEI dataset
                ms = [-3.01, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.01]
                input_tensor = torch.clip(input_tensor,min=-3., max=3.)
            res = input_tensor.clone()
            for i in range(self.queue_patch):
                res[torch.logical_and(input_tensor > ms[i],input_tensor <= ms[i+1])]=i
        else:  # for classification
            res = input_tensor
        return res

    def empty_queue(self):
        self.queue_ptr = torch.zeros(self.output_dim, dtype=torch.long)
        self.queue_label = torch.randn(self.queue_len.sum().item(), 1)
        self.queue_cla_label = self.queue_label.clone()
    
    def dequeue_and_enqueue(self, feats, labels,  preds=None):
        
        # feats: [batchsize, dimension]
        # labels: [batchsize]
        # preds: [batchsize, num_class] one_hot
        if self.is_empty:
            return

        cla_labels = self.logit2cla(labels)
        ### choose TP
        if preds != None:
            preds = self.logit2cla(preds)
            keep = preds == cla_labels

            feats = feats[keep]
            labels = labels[keep]
        # print(labels)
        for patch in range(self.queue_patch):
            indices_per_class = torch.nonzero(cla_labels==patch).squeeze(-1)
            nums_per_class = indices_per_class.shape[0]
            # print(patch,":",nums_per_class,':',self.queue_ptr[patch])
            if nums_per_class == 0:
                continue
            feat_per_class = feats[indices_per_class]
            # print(feat_per_class)
            label_per_class = labels[indices_per_class].unsqueeze(-1)
            cla_label_per_class = cla_labels[indices_per_class].unsqueeze(-1)
            ptr = int(self.queue_ptr[patch])
            start =  self.queue_start[patch]
            ptr = start + ptr
            end = self.queue_end[patch]
            if ptr + nums_per_class >= end:
                self.queue[ptr:end,:] = feat_per_class[:end-ptr,:]
                self.queue[start:start + nums_per_class + ptr - end ,:] = feat_per_class[end-ptr:,:]
                
                self.queue_label[ptr:end] = label_per_class[:end - ptr]
                self.queue_label[start:start + nums_per_class + ptr - end] = label_per_class[end-ptr:]

                self.queue_cla_label[ptr:end] = cla_label_per_class[:end - ptr]
                self.queue_cla_label[start:start + nums_per_class + ptr - end] = cla_label_per_class[end-ptr:]

            else:
                self.queue[ptr:ptr+nums_per_class, :] = feat_per_class
                self.queue_label[ptr:ptr + nums_per_class] = label_per_class
                self.queue_cla_label[ptr:ptr + nums_per_class] = cla_label_per_class
            ptr = (int(self.queue_ptr[patch]) + nums_per_class) %  (end-start)  # move pointer
            self.queue_ptr[patch] = ptr
    def print_queue(self):
        for patch in range(self.queue_patch):
            print(patch,":")
            ptr = int(self.queue_ptr[patch])
            start =  self.queue_start[patch]
            ptr = start + ptr
            print("queue_label:", self.queue_label[start:ptr+1].squeeze(-1))
            print("queue_cla_label:", self.queue_cla_label[start:ptr+1].squeeze(-1))

    def get_queue(self, device=None):
        feat = self.queue.clone().detach()
        label = self.queue_label.clone().detach()
        cla_label = self.queue_cla_label.clone().detach()
        if device !=None:
            feat = feat.to(device)
            label = label.squeeze(-1).to(device)
            cla_label = cla_label.squeeze(-1).to(device)
        return feat, label, cla_label
    
    def learning_loss(self, feat, label):
        # if self.queue_patch<=3:
        #     norm_feat = F.normalize(feat, dim=-1)
        # else:
        #     norm_feat = feat
        device = feat.device
        if self.is_empty:
            total_feat = feat
            total_label = label
            total_cla_label = self.logit2cla(label)
        else:
            queue_feat, queue_label, queue_cla_label = self.get_queue(device = device)
            
            total_feat = torch.cat((queue_feat, feat), dim=0)
            total_label = torch.cat((queue_label, label), dim=0)
            total_cla_label = torch.cat((queue_cla_label, self.logit2cla(label)), dim=0)
        if self.criterion_loss in ['uacccon', 'acccon']:
            # print(total_cla_label)
            # print(total_label)
            loss =  self.con_loss(total_feat,total_cla_label, total_label)
        else:
            loss =  self.con_loss(total_feat,total_cla_label)

        return loss
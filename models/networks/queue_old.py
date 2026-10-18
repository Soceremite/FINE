import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.utils.functions_old import SupConLoss,UACCConLoss,ACCConLoss,RnCLoss
class FeatQueue(nn.Module):
    """
    Dynamic Contrastive Queue
    get_queue: get all features and label
    """
    def __init__(self, num_classes, samples_per_class, 
                 min_queue_len, queue_ratio, hidden_size,
                 temperature, criterion_loss='supcon'):
        super(FeatQueue, self).__init__()

        self.queue_ratio = queue_ratio
        self.min_queue_len = min_queue_len
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
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        
        # 初始化
        self.queue = F.normalize(self.queue, dim=0)
        criterion_loss = criterion_loss.lower()
        if criterion_loss =='uacccon':
            print('FeatQueue Loss','UACCConLoss')
            self.con_loss = UACCConLoss(label_scale=num_classes,
                                       temperature=temperature)
        elif criterion_loss == 'acccon':
            print('FeatQueue Loss','ACCConLoss')
            self.con_loss = ACCConLoss(label_scale=num_classes,
                                       temperature=temperature)
        elif criterion_loss == 'rnc':
            print('FeatQueue Loss','RnCLoss')
            self.con_loss = RnCLoss()
        else:
            print('FeatQueue Loss','SupConLoss')
            self.con_loss = SupConLoss(temperature=temperature)
    
    def reg2cla(self,input_tensor):
        '''
        regression to classification
        input:[batchsize, 1]
        '''
        if self.queue_patch==2:
            ms = [-1.01, 0.0, 1.01]
            input_tensor = torch.clip(input_tensor,min=-1., max=1.)
        elif self.queue_patch==3:
            ms = [-1.01,-0.5, 0.5, 1.01]
            input_tensor = torch.clip(input_tensor,min=-1., max=1.)
        elif self.queue_patch==5:
            ms = [-2.01, -1.5, -0.5, 0.5, 1.5, 2.01]
            input_tensor = torch.clip(input_tensor,min=-2., max=2.)
        elif self.queue_patch==7:
            ms = [-3.01, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.01]
            input_tensor = torch.clip(input_tensor,min=-3., max=3.)
        res = input_tensor.clone()
        for i in range(self.queue_patch):
            res[torch.logical_and(input_tensor > ms[i],input_tensor <= ms[i+1])]=i
        return res

    def empty_queue(self):
        self.queue_ptr = torch.zeros(self.output_dim, dtype=torch.long)
        self.queue_label = torch.randn(self.queue_len.sum().item(), 1)
    
    def dequeue_and_enqueue(self, feats, labels,  preds=None):
        
        # if self.queue_patch<=3:
        #     feats = F.normalize(feats, dim=-1)
        # feats: [batchsize, dimension]
        # labels: [batchsize]
        # preds: [batchsize, num_class] one_hot
        if self.is_empty:
            return

        labels = self.reg2cla(labels)
        ### choose TP
        if preds != None:
            preds = self.reg2cla(preds)
            keep = preds == labels

            feats = feats[keep]
            labels = labels[keep]
        # print(labels)
        for patch in range(self.queue_patch):
            indices_per_class = torch.nonzero(labels==patch).squeeze(-1)
            nums_per_class = indices_per_class.shape[0]
            # print(patch,":",nums_per_class,':',self.queue_ptr[patch])
            if nums_per_class == 0:
                continue
            feat_per_class = feats[indices_per_class]
            # print(feat_per_class)
            label_per_class = labels[indices_per_class].unsqueeze(-1)
            
            ptr = int(self.queue_ptr[patch])
            start =  self.queue_start[patch]
            ptr = start + ptr
            end = self.queue_end[patch]
            if ptr + nums_per_class >= end:
                self.queue[ptr:end,:] = feat_per_class[:end-ptr,:]
                self.queue[start:start + nums_per_class + ptr - end ,:] = feat_per_class[end-ptr:,:]
                self.queue_label[ptr:end] = label_per_class[:end - ptr]
                self.queue_label[start:start + nums_per_class + ptr - end] = label_per_class[end-ptr:]
            else:
                self.queue[ptr:ptr+nums_per_class, :] = feat_per_class
                self.queue_label[ptr:ptr + nums_per_class] = label_per_class
            ptr = (ptr + nums_per_class) %  (end-start)  # move pointer
            self.queue_ptr[patch] = ptr
            
    def get_queue(self, device=None):
        feat = self.queue.clone().detach()
        label = self.queue_label.clone().detach()
        if device !=None:
            feat = feat.to(device)
            label = label.squeeze(-1).to(device)
        return feat, label
    
    def learning_loss(self, feat, label):
        # if self.queue_patch<=3:
        #     norm_feat = F.normalize(feat, dim=-1)
        # else:
        #     norm_feat = feat
        device = feat.device
        if self.is_empty:
            total_feat = feat
            total_label = label
        else:
            queue_feat, queue_label = self.get_queue(device = device)
            label = self.reg2cla(label)
            total_feat = torch.cat((queue_feat, feat), dim=0)
            total_label = torch.cat((queue_label, label), dim=0)
        loss =  self.con_loss(total_feat, total_label)

        return loss
from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import json
import numpy as np
"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

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


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)  # 按维度求均值，keepdims=True保持转换后维度不变
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()  # 求范数
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
class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds:softmax输出结果
        labels:真实值
        """

        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        # negatives_mask = 1. - mask
        negatives_mask = logits_mask - positives_mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class ACCConLoss(nn.Module):

    def __init__(self, 
                 label_scale=7.0, 
                 temperature=0.05, 
                 scale_by_temperature=True,
                 epsilon=1e-6):
        super(ACCConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.label_scale=label_scale
        self.epsilon = epsilon

    def forward(self, features, labels):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size]
        输出:
            loss值
        """
        # np.save('feat.npy', features.detach().cpu().numpy())
        # print(labels)
        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        # mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positive_mask = mask * logits_mask
        negative_mask = logits_mask - positive_mask
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        positive_exp_logits = torch.exp(logits)

        labels = labels.squeeze(-1)
        phi = torch.pi *(1-(labels.unsqueeze(0)-labels.unsqueeze(-1))/self.label_scale)
        sim = torch.matmul(features, features.T)-torch.eye(batch_size).to(device)
        # print(phi)
        # print(phi.shape)
        # print(sim)
        # print(sim.shape)
        # print("torch.abs(torch.sin(phi))",torch.abs(torch.sin(phi)))
        # print(self.epsilon)
        # print("torch.sqrt(1-sim**2+self.epsilon)",torch.sqrt(torch.abs(1.0-sim**2)+self.epsilon))
        # print("torch.abs(torch.sin(phi))*torch.sqrt(1-sim**2+self.epsilon)",torch.abs(torch.sin(phi))*torch.sqrt(1-sim**2+self.epsilon))
        negative_logits = torch.div(
            sim* torch.cos(phi)-torch.abs(torch.sin(phi))*torch.sqrt(1.0-sim**2+self.epsilon),
            self.temperature)
        # print("negative_logits:",negative_logits)
        negative_exp_logits = torch.exp(negative_logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        num_positive_per_row = torch.sum(positive_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            negative_exp_logits * negative_mask, axis=1, keepdims=True) + torch.sum(
            positive_exp_logits * positive_mask, axis=1, keepdims=True)
        # print("negative_mask:",negative_mask)
        # print("positive_mask:",positive_mask)
        # print("negative_exp_logits:",negative_exp_logits)
        # print("positive_exp_logits:",positive_exp_logits)
        # print(denominator.shape)
        # print(denominator)
        #因为后面会乘positive_mask 所以这里可以直接用logits
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positive_mask, axis=1)[num_positive_per_row > 0] / num_positive_per_row[
                        num_positive_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss



class UACCConLoss(nn.Module):

    def __init__(self, 
                 label_scale=7.0, 
                 temperature=0.05,
                 bias_sigma=0.2,
                 alpha=1.5,
                 scale_by_temperature=True,
                 epsilon=1e-6):
        super(UACCConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.label_scale=label_scale
        self.epsilon = epsilon
        self.bias_sigma = bias_sigma
        self.alpha = alpha

    def forward(self, features, labels):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size]
        输出:
            loss值
        """
        # np.save('feat.npy', features.detach().cpu().numpy())
        # print(labels)
        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        # mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positive_mask = mask * logits_mask
        negative_mask = logits_mask - positive_mask
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        positive_exp_logits = torch.exp(logits)

        labels = labels.squeeze(-1) # [0-6]
        phi = torch.pi *(1-(labels.unsqueeze(0)-labels.unsqueeze(-1))/self.label_scale)

        labels = labels-self.label_scale//2   #[-3,3]
        s = labels.unsqueeze(0)*labels.unsqueeze(-1)
        s1 = torch.where(s>0,1.0,torch.where(s<0,-1.0,s)) #[-1,0,1]

        phi = phi*(1+s1*self.bias_sigma)**self.alpha

        sim = torch.matmul(features, features.T)-torch.eye(batch_size).to(device)

        negative_logits = torch.div(
            sim* torch.cos(phi)-torch.abs(torch.sin(phi))*torch.sqrt(1.0-sim**2+self.epsilon),
            self.temperature)
        # print("negative_logits:",negative_logits)
        negative_exp_logits = torch.exp(negative_logits)

        num_positive_per_row = torch.sum(positive_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            negative_exp_logits * negative_mask, axis=1, keepdims=True) + torch.sum(
            positive_exp_logits * positive_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positive_mask, axis=1)[num_positive_per_row > 0] / num_positive_per_row[
                        num_positive_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # features: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, feat_dim]
        # labels: [bs, label_dim]

        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = labels.repeat(2, 1)  # [2bs, label_dim]

        labels = labels.unsqueeze(-1)
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, emo_sample_per_class=None, int_sample_per_class=None):
        super(BalancedSoftmax, self).__init__()
        self.emo_sample_per_class = emo_sample_per_class
        self.int_sample_per_class = int_sample_per_class


    def forward(self, input, label, reduction="mean"):
        if input.shape[1]==7:
            return balanced_softmax_loss(label, input, self.emo_sample_per_class, reduction)
        elif input.shape[1]==8:
            return balanced_softmax_loss(label, input, self.int_sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss



class FCC(nn.Module):
    """
    Feature clusters compression (FCC)
    gamma: the hyper-parameter for setting scaling factor tau.
    c_type: compression type:
                'edc' is equal difference compression.
    """
    def __init__(self, emo_sample_per_class=None, int_sample_per_class=None,gamma=0.5):
        super(FCC,self).__init__()

        self.emo_num_classes = len(emo_sample_per_class)
        self.int_num_classes = len(int_sample_per_class)

        # self.emo_sorted_indices = np.argsort(emo_sample_per_class)
        # self.int_sorted_indices = np.argsort(int_sample_per_class)
        # emo_tau = []
        # for k in range(self.emo_num_classes):
        #     emo_tau.append(round((1 + gamma - k*(gamma/self.emo_num_classes)),2))
        # self.emo_tau = [0 for i in range(self.emo_num_classes)]
        
        # for i, key in enumerate(self.emo_sorted_indices):
        #     self.emo_tau[key]= emo_tau[self.emo_num_classes-i-1]
        #     # self.emo_tau[key]= emo_tau[i]
        # print(self.emo_tau)
        # int_tau = []
        # for k in range(self.int_num_classes):
        #     int_tau.append(round((1 + gamma - k*(gamma/self.int_num_classes)),2))
        # self.int_tau = [0 for i in range(self.int_num_classes)]
        
        # for i, key in enumerate(self.int_sorted_indices):
        #     self.int_tau[key]= int_tau[self.int_num_classes-i-1]
        #     # self.int_tau[key]= int_tau[i]
        # print(self.int_tau)
        emo_sample_per_class = torch.tensor(emo_sample_per_class)
        self.emo_tau = emo_sample_per_class / torch.sum(emo_sample_per_class) + 1
        
        int_sample_per_class = torch.tensor(int_sample_per_class)
        self.int_tau = int_sample_per_class / torch.sum(int_sample_per_class) + 1
        print(self.emo_tau)
        print(self.int_tau)
        

    def equal_diff_compress(self, feature, label, tau):

        bs = feature.shape[0]
        tau_batch = []
        for j in range(bs):
            tau_batch.append(tau[label[j]])
        # setting scaling factor tau
        raw_shape = feature.shape
        device = feature.device
        tau_batch = torch.tensor(tau_batch).to(device)

        tau_batch = tau_batch.view(bs, 1)
        feature = feature.view(bs, -1)

        new_features = torch.mul(feature, tau_batch)
        new_features = new_features.view(raw_shape)

        return new_features

        
    def forward(self, feature, label):
        
        # compressing feature
        if feature.shape[1]==self.emo_num_classes:
            tau = self.emo_tau
        else:
            tau = self.int_tau
        new_features = self.equal_diff_compress(feature, label, tau)
        return new_features


def get_logits( image_features, text_features, logit_scale):
    # 计算image_features @ text_features.T相似度矩阵
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T
    return logits_per_image, logits_per_text

def cal_clip_loss(image_features, text_features, logit_scale=0.07):
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return total_loss

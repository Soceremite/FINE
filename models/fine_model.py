import torch
import os
import json
import torch.nn.functional as F
import torch.nn as nn
from models.networks.Qformer import MoQformer
from models.base_model import BaseModel
from models.networks.queue import FeatQueue
from models.utils.config import OptConfig
from models.networks.util_module import Modal_Decoder, BaseClassifier
from models.networks.FTREncoder import FTREncoder
from models.networks.modal_encoder import BertEncoder
import warnings
# 捕获警告并显示堆栈信息
warnings.simplefilter("always")  # 设置总是显示警告
warnings.filterwarnings('default', category=UserWarning)  # 确保警告会被触发


import math
import numpy as np
torch.autograd.set_detect_anomaly(True)

class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory):
        mem = self.norm(memory)
        return mem

def loss_func(name='mse'):
    loss_map = {
        'mse': nn.MSELoss,
        'l1': nn.L1Loss,
        'ce': nn.CrossEntropyLoss
    }
    return loss_map[name]()
 
class finemodel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--embd_size_a', default=512, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_t', default=512, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=512, type=int, help='visual model embedding size')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--num_group', type=int, default=4)
        parser.add_argument('--topk_ratio', type=float, default=0.75)  
        parser.add_argument('--query_len', type=int, default=64)
        parser.add_argument('--min_queue_len', type=int, default=64)
        parser.add_argument('--queue_ratio', type=float, default=0.5)
        parser.add_argument('--criterion_loss', type=str, default ='mse',help='which loss to use')
        parser.add_argument('--multi_weight', type=float, default=1.0)
        parser.add_argument('--uni_weight', type=float, default=1.0)
        parser.add_argument('--aux_weight', type=float, default=1.0)
        parser.add_argument('--cl_weight', type=float, default=1.0)
        parser.add_argument('--ds_weight', type=float, default=1.0)
        parser.add_argument('--sha_weight', type=float, default=1.0)
        parser.add_argument('--dec_weight', type=float, default=1.0)
        parser.add_argument('--recon_weight', type=float, default=1.0)
        parser.add_argument('--tu_weight', type=float, default=1.0)
        parser.add_argument('--sha_ir_weight', type=float, default=1.0)
        
        parser.add_argument('--temperature', type=float, default=0.07, help='temperature of contrastive learning loss')
        parser.add_argument('--activate_fun', type=str, default='relu', help='which activate function will be used')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        # parameter of Transformer Encoder
        parser.add_argument('--Modal_head', type=int, default=2, help='head of Transformer_head')
        parser.add_argument('--Modal_layers', type=int, default=1, help='head of Transformer_head')
        parser.add_argument('--Fusion_head', type=int, default=2, help='head of Transformer_head')
        parser.add_argument('--Fusion_layers', type=int, default=1, help='layer of Transformer_head')
        # parameter of multi-head attention
        parser.add_argument('--attention_head', type=int, default=1, help='head of multi-head attention')
        parser.add_argument('--attention_dropout', type=float, default=0., help='head of multi-head attention')
        # other parameter
        parser.add_argument('--drop_last', type=bool, default=False, help='drop the last data or not')

        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = []
        self.model_names = []  # 所有模块的名称
        self.train_mode = opt.train_mode

        self.netNormT = Normalize(opt.input_dim_t)
        self.model_names.append('NormT')
        self.netNormV = Normalize(opt.input_dim_v)
        self.model_names.append('NormV')
        self.netNormA = Normalize(opt.input_dim_a)
        self.model_names.append('NormA')

        
        self.netBERT = BertEncoder(finetune=opt.bert_finetune,
                                   pretrain=opt.bert_pretrain)
        self.model_names.append('BERT')

        # lexical model
        self.netMoQformerT = MoQformer(input_dim=opt.input_dim_t,
                                       qformer_heads = opt.qformer_heads,
                                       qformer_layers = opt.qformer_layers,
                                       num_experts=opt.num_group,
                                       topk_ratio=opt.topk_ratio,
                                       encoder_width=opt.embd_size_t)
        self.model_names.append('MoQformerT')

        # visual model
        self.netMoQformerV = MoQformer(input_dim=opt.input_dim_v,
                                       qformer_heads = opt.qformer_heads,
                                       qformer_layers = opt.qformer_layers,
                                       num_experts=opt.num_group,
                                       topk_ratio=opt.topk_ratio,
                                       encoder_width=opt.embd_size_v)
        self.model_names.append('MoQformerV')

        # acoustic model
        self.netMoQformerA = MoQformer(input_dim=opt.input_dim_a,
                                       qformer_heads = opt.qformer_heads,
                                       qformer_layers = opt.qformer_layers,
                                       num_experts=opt.num_group,
                                       topk_ratio=opt.topk_ratio,
                                       encoder_width=opt.embd_size_a)
        self.model_names.append('MoQformerA')

        self.netTREncoder = FTREncoder(input_dim=[opt.embd_size_a, opt.embd_size_t, opt.embd_size_v], 
                                      shauni_dim=opt.shauni_dim,
                                      shauni_tr_dim=opt.shauni_tr_dim,
                                      task_dim=opt.shauni_task_dim,
                                      mi_hidden_dim=opt.mi_hidden_dim,
                                      dropout=opt.dropout_rate,
                                      optimizer_mode=opt.optimizer_mode,
                                      weight_decay=opt.weight_decay,
                                      activation=opt.activate_fun,
                                      mi_lr=opt.mi_lr
                                      )
        self.model_names.append('TREncoder')


        self.task_hidden_dim = [ item*2 for item in opt.shauni_tr_dim]

        print("self.task_hidden_dim",self.task_hidden_dim)

        self.netModalDecoder = nn.ModuleList([
            Modal_Decoder(opt.output_dim, self.task_hidden_dim[i], 
                          nhead=opt.Modal_head, 
                          ffn_dim=self.task_hidden_dim[i]*2, 
                          dropout=opt.dropout_rate, 
                          num_layers=opt.Modal_layers, 
                          activation=opt.activate_fun,
                          train_mode=self.train_mode) for i in range(3)])
        self.model_names.append('ModalDecoder')

        # Transformer Fusion model
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sum(self.task_hidden_dim),
                                                         nhead=int(opt.Fusion_head),
                                                         batch_first=True)
        self.netFusion = torch.nn.TransformerEncoder(encoder_layer, num_layers=opt.Fusion_layers, norm=nn.LayerNorm(sum(self.task_hidden_dim)))
        self.model_names.append('Fusion')


        self.temperature = opt.temperature
        self.cls_input_size = 2 * sum(self.task_hidden_dim)
        self.num_classes = opt.num_classes

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        self.netFCF = BaseClassifier(self.cls_input_size, cls_layers, output_size=opt.output_dim, activation=opt.activate_fun)
        self.model_names.append('FCF')
        print(self.netFCF)

        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        if self.train_mode == "classification":
            self.softmax = nn.Softmax(dim=1)

        self.loss_names.append('MM')
        self.loss_names.append('UM')
        self.loss_names.append('M_CL')
        self.loss_names.append('AUX')
        self.loss_names.append('SHA')
        self.loss_names.append('UNI')
        self.loss_names.append('RECON')
        self.loss_names.append('UNI_TR')
        self.loss_names.append('SHA_TI')

        if self.isTrain:

            self.queue = FeatQueue(self.num_classes, opt.samples_per_class, 
                               opt.min_queue_len, opt.queue_ratio, self.cls_input_size,
                               self.temperature, self.train_mode, criterion_loss=opt.con_loss,
                               min_label=opt.min_label, max_label=opt.max_label)

            self.load_pretrained_encoder(opt)
            
            self.criterion_loss = loss_func(opt.criterion_loss)

            self.unimodal_loss = loss_func(opt.criterion_loss)
            
            self.recon_mse = nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            opti_model = self.model_names.copy()
            # opti_model.remove('CLUB')
            opti_model.remove('BERT')
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in opti_model]
            
            bert_param = list(self.netBERT.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_grouped_parameters = [
                {'params': [p for n, p in bert_param
                            if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
                {'params': [p for n, p in bert_param
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            
            
            if opt.optimizer_mode =="adamw":
                 self.optimizer = torch.optim.AdamW(paremeters, 
                                          lr=opt.lr, 
                                          betas=(opt.beta1, 0.999),
                                          weight_decay=opt.weight_decay)
                 self.bert_optimizer = torch.optim.AdamW(bert_grouped_parameters,
                                                         lr=opt.bert_lr)
            else:
                self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=opt.weight_decay)
                self.bert_optimizer = torch.optim.Adam(bert_grouped_parameters, lr=opt.bert_lr,weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.bert_optimizer)
            self.optimizers.extend(self.netTREncoder.get_optimizers())
            # self.optimizers.append(self.emoclub_optimizer)
            self.output_dim = opt.output_dim
            self.uni_weight = opt.uni_weight
            self.multi_weight = opt.multi_weight
            self.cl_weight = opt.cl_weight
            self.aux_weight = opt.aux_weight
            self.sha_weight = opt.sha_weight
            self.dec_weight = opt.dec_weight
            self.recon_weight = opt.recon_weight
            self.tu_weight = opt.tu_weight
            self.sha_ir_weight = opt.sha_ir_weight

            self.nums_modal = 3.
        # else:
        #     self.load_pretrained_encoder(opt)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # 加载预训练Encoder，
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format('None'))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

        
    
    def get_mask_from_length(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()


        batch_size = lengths.shape[0]
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(self.device)
        # bool
        mask =ids < lengths.unsqueeze(1).expand(-1, max_len)
        # 0,1
        mask = torch.where(mask, 1.0,0.0).long()
        return mask
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.audio = input['audio'].float().to(self.device)
        # self.raw_text = input['raw_text']
        self.input_ids = input['input_ids'].to(self.device)
        self.attention_mask = input['attention_mask'].to(self.device)
        # self.text = input['text'].float().to(self.device)
        self.visual = input['vision'].float().to(self.device)
        
        ### mask
        self.audio_mask = input['audio_lengths'].float().to(self.device)
        self.visual_mask = input['vision_lengths'].float().to(self.device)

        if self.train_mode =='regression':
            self.label = input['regression_labels'].float().to(self.device)
        else:
            self.label = input['classification_labels'].float().to(self.device)

    def emb_avgpool(self, feat_embedding):
        feat_embedding = feat_embedding.transpose(1,2)
        feat_embedding = self.avgpool1d(feat_embedding)
        feat_embedding = feat_embedding.squeeze(-1)
        return feat_embedding
    

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion

        a_atts = self.get_mask_from_length(self.audio_mask, max_len=self.audio.shape[1])
        v_atts = self.get_mask_from_length(self.visual_mask, max_len=self.visual.shape[1])

        self.text = self.netBERT(input_ids=self.input_ids,
                                 attention_mask = self.attention_mask)
        
        if self.isTrain:
            self.feat_Tq, self.loss_AUX_T = self.netMoQformerT(self.text, isTrain=self.isTrain)
            # print("Qformer T output:", self.feat_Tq.shape)
            self.feat_Vq, self.loss_AUX_V = self.netMoQformerV(self.visual, v_atts, isTrain=self.isTrain)
            # print("Qformer V output:", self.feat_Vq.shape)
            self.feat_Aq, self.loss_AUX_A = self.netMoQformerA(self.audio, a_atts, isTrain=self.isTrain)
            # print("Qformer A output:", self.feat_Aq.shape)
            self.feat_a_str, self.feat_t_str, self.feat_v_str, \
            self.feat_a_utr, self.feat_t_utr, self.feat_v_utr, \
            self.loss_share, self.loss_unique, self.loss_recon,\
            self.loss_sha_ti, self.loss_uni_tr = self.netTREncoder(self.feat_Aq, self.feat_Tq, self.feat_Vq, self.label)
        else:
            self.feat_Tq = self.netMoQformerT(self.text, isTrain=self.isTrain)
            self.feat_Vq = self.netMoQformerV(self.visual, v_atts, isTrain=self.isTrain)
            self.feat_Aq = self.netMoQformerA(self.audio, a_atts, isTrain=self.isTrain)
            self.feat_a_str, self.feat_t_str, self.feat_v_str, \
            self.feat_a_utr, self.feat_t_utr, self.feat_v_utr = self.netTREncoder(self.feat_Aq, self.feat_Tq, self.feat_Vq)
        # print("feat_t_sha", self.feat_t_sha.shape)
        # print("feat_t_uni", self.feat_t_uni.shape)
        # print("feat_v_sha", self.feat_v_sha.shape)
        # print("feat_v_uni", self.feat_v_uni.shape)
        # print("feat_a_sha", self.feat_a_sha.shape)
        # print("feat_a_uni", self.feat_a_uni.shape)
        
        self.feat_a_task = torch.cat([self.feat_a_str, self.feat_a_utr], dim=-1)
        self.feat_t_task = torch.cat([self.feat_t_str, self.feat_t_utr], dim=-1)
        self.feat_v_task = torch.cat([self.feat_v_str, self.feat_v_utr], dim=-1)

        # print(self.feat_t_tr.shape)

        self.unimodal_pred_a, self.cls_a = self.netModalDecoder.module[0](self.feat_a_task)
        self.unimodal_pred_t, self.cls_t = self.netModalDecoder.module[1](self.feat_t_task)
        self.unimodal_pred_v, self.cls_v = self.netModalDecoder.module[2](self.feat_v_task)

        # print(self.cls_decoder_T.shape)

        # # fusion module
        fusion_feat = torch.cat((self.feat_a_task, self.feat_t_task, self.feat_v_task), dim=-1)
        fusion_feat = self.netFusion(fusion_feat)

        fusion_feat = self.emb_avgpool(fusion_feat)
        
        self.fusion_feat = torch.cat((fusion_feat, self.cls_a, self.cls_t, self.cls_v),dim=1)
        
        self.logits_fusion = self.netFCF(self.fusion_feat)
        if self.train_mode == 'regression':
            self.pred = self.logits_fusion.squeeze(-1)
        else:
            self.pred = self.softmax(self.logits_fusion)
    
    def backward(self):
        """Calculate the loss for back propagation"""

        self.loss_AUX = self.aux_weight * (self.loss_AUX_T + self.loss_AUX_V + self.loss_AUX_A) / self.nums_modal


        self.loss_SHA = self.sha_weight * self.loss_share / self.nums_modal

        self.loss_UNI = self.dec_weight * self.loss_unique / self.nums_modal
        
        self.loss_RECON = self.recon_weight * self.loss_recon / self.nums_modal

        self.loss_UNI_TR =  self.tu_weight * self.loss_uni_tr / self.nums_modal

        self.loss_SHA_TI = self.sha_ir_weight * self.loss_sha_ti / self.nums_modal

        if self.train_mode == 'classification':
            label = self.label.long()
        else:
            label = self.label
        loss_UM = self.unimodal_loss(self.unimodal_pred_a, label) \
                          + self.unimodal_loss(self.unimodal_pred_t, label) \
                          + self.unimodal_loss(self.unimodal_pred_v, label)
        self.loss_UM = self.uni_weight * loss_UM / self.nums_modal

        self.loss_MM = self.multi_weight * self.criterion_loss(self.pred, label)
        
        self.norm_fusion_feat = F.normalize(self.fusion_feat, dim=-1)
        self.loss_M_CL = self.cl_weight * self.queue.learning_loss(self.norm_fusion_feat,self.label)

        # self.queue.dequeue_and_enqueue(self.norm_fusion_feat.clone().detach(), self.label.clone().detach(), self.pred.clone().detach())
        self.queue.dequeue_and_enqueue(self.norm_fusion_feat.clone().detach(), self.label.clone().detach())
        

        loss = self.loss_MM + self.loss_UM \
               + self.loss_M_CL + self.loss_AUX \
               + self.loss_SHA + self.loss_UNI + self.loss_RECON  \
               + self.loss_UNI_TR + self.loss_SHA_TI
               
        # backward
        self.optimizer.zero_grad()
        self.bert_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.bert_optimizer.step()

        ### mi
        self.netTREncoder.module.uni_backward(mm_iter=3)
        self.netTREncoder.module.sti_backward(mm_iter=3)

        # for model in self.model_names:
        #     torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # print('forward has done')
        self.backward()
        # print('backward has done')
    
    def get_criterion_loss(self, pred, label):
        return self.criterion_loss(pred, label)
        

class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
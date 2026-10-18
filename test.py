import os
import time
import numpy as np
from get_opts import Options, save_json
from dataloader import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from utils.metrics import Metrics
from utils.util import set_random_seed,plot_confusion_matrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import torch
from torch.nn import MSELoss
from random import random
from collections import OrderedDict
from models.networks import tools
import pickle
import math
import h5py
from tqdm import tqdm

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def eval(model, val_iter, metrics, phase='val', batch_size = 64):
    model.eval()
    total_pred = []
    total_label = []
    fusion_feat =[]
    audio_str_feat = []
    text_str_feat = []
    vision_str_feat = []
    audio_utr_feat = []
    text_utr_feat = []
    vision_utr_feat = []

    num_batches = math.ceil(len(val_iter) / batch_size)

    for i, data in tqdm(enumerate(val_iter), total=num_batches, ncols=100):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.detach().cpu()
        label = data['regression_labels'].detach().cpu()
        total_pred.append(pred)
        total_label.append(label)
        fusion_feat.append(model.fusion_feat.detach().cpu().numpy())
        audio_str_feat.append(model.feat_a_str.detach().cpu().numpy())
        text_str_feat.append(model.feat_t_str.detach().cpu().numpy())
        vision_str_feat.append(model.feat_v_str.detach().cpu().numpy())
        audio_utr_feat.append(model.feat_a_utr.detach().cpu().numpy())
        text_utr_feat.append(model.feat_t_utr.detach().cpu().numpy())
        vision_utr_feat.append(model.feat_v_utr.detach().cpu().numpy())
    
    total_pred = torch.cat(total_pred,dim=0)
    total_label = torch.cat(total_label, dim=0)
    eval_res = metrics(total_pred, total_label)
    
    fusion_feat =np.concatenate(fusion_feat, axis=0)
    audio_str_feat = np.concatenate(audio_str_feat, axis=0)
    text_str_feat = np.concatenate(text_str_feat, axis=0)
    vision_str_feat = np.concatenate(vision_str_feat, axis=0)
    audio_utr_feat = np.concatenate(audio_utr_feat, axis=0)
    text_utr_feat = np.concatenate(text_utr_feat, axis=0)
    vision_utr_feat = np.concatenate(vision_utr_feat, axis=0)
    labels = total_label.numpy()

    h5_path= os.path.join(model.log_dir, 'hidden_feat.h5')
    with h5py.File(h5_path,'w') as f:
        f.create_dataset('labels',data = labels)
        f.create_dataset('fusion_feat',data = fusion_feat)
        f.create_dataset('audio_str_feat',data = audio_str_feat)
        f.create_dataset('text_str_feat',data = text_str_feat)
        f.create_dataset('vision_str_feat',data = vision_str_feat)
        f.create_dataset('audio_utr_feat',data = audio_utr_feat)
        f.create_dataset('text_utr_feat',data = text_utr_feat)
        f.create_dataset('vision_utr_feat',data = vision_utr_feat)

    return eval_res


def move_model_to_device(model, opt):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        net = tools.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
        setattr(model, 'net' + name, net)

if __name__ == '__main__':
    opt = Options().parse(isTrain = False)  # get training options
    
    test_dataset = create_dataset_with_args(opt, type=['test'])
    
    # opt.num_group = [1, 2, 4, 8, 12, 16, 24, 32]
    opt.num_group = [opt.query_token_len for _ in range(opt.num_experts)]

    dataparams =  test_dataset._get_dataparams()
    for key, value in dataparams.items():
        setattr(opt, key, value)
    
    model = create_model(opt)  # create a model given opt.model and other options
    
    gpu_id = int(opt.gpu_ids[0])

    device = torch.device(f'cuda:{gpu_id}' if gpu_id else 'cpu')
    move_model_to_device(model, opt)
    checkpoint_path = os.path.join(opt.checkpoints_dir, 'weight')
    model.load_networks_epoch(checkpoint_path, opt.best_epoch)
    metrics = Metrics(train_mode=opt.train_mode).getMetics(opt.dataset_name)

    test_result = eval(model, test_dataset, metrics, phase='test', batch_size = opt.batch_size)
    print('Test result ')
    for key, val in test_result.items():
       print('test_%s:%.4f' %(key,val))
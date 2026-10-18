import torch
import numpy as np
import random
import os
import json
import _pickle as pk
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from dataloader.base_dataset import BaseDataset
from models.networks.modal_encoder import bert_tokenizer

class VADDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, type):
        ''' VAD dataset reader
            type in ['train', 'valid', 'test']
        '''
        dtype_dic = {'train':'train',
                     'valid':'dev',
                     'test':'test'}
        self.dataset_name  = opt.dataset_name
        self.dataset_type = dtype_dic[type]

        self.cur_experiment = 'V'

        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        with open(os.path.join(pwd, 'config', f'{self.dataset_name}_config.json'), 'r', encoding='UTF-8') as f:
            self.config = json.load(f)
        if opt.aligned:
            data_path = self.config['align_data_path']
        else:
            data_path = self.config['raw_data_path']
        print("loading from: ", data_path)
        self.samples_per_class = self.config[f'samples_per_class_{str(opt.num_classes)}_{self.cur_experiment}']

        audio_feature_path = os.path.join(data_path, "features/audio_feature.pkl")
        danmu_feature_path = os.path.join(data_path, "features/danmu_feature.pkl")
        vision_feature_path = os.path.join(data_path, "features/visual_feature.pkl")

        with open(audio_feature_path, "rb") as f:
            self.audio_feature = pk.load(f)
        with open(danmu_feature_path, "rb") as f:
            self.danmu_feature = pk.load(f)
        with open(vision_feature_path, "rb") as f:
            self.vision_feature = pk.load(f)


        with open(os.path.join(data_path, f'new_split_labels/independence/cv_2_independence/{self.cur_experiment}_{self.dataset_type}.json')) as f:
            data = json.load(f)
            self.segment_id = list(data.keys())
            self.segment_label = list(data.values())
        assert len(self.segment_id) == len(self.segment_label)
        print("---对音频和视觉特征进行检测：是否有Video ID未出现在特征中---")
        for each_id in self.segment_id:
            if each_id not in self.audio_feature.keys():
                print(each_id, "音频中没有相应ID")

    def _get_dataparams(self):

        return  {
            "samples_per_class":self.samples_per_class,
            "min_label":self.config[f'min_label_{self.cur_experiment}'],
            "max_label":self.config[f'max_label_{self.cur_experiment}'],
            "input_dim_t":self.config['input_dim_t'],
            "input_dim_v":self.config['input_dim_v'],
            "input_dim_a":self.config['input_dim_a']
        }
    

    def get_data(self):
        return self.segment_id

    def __len__(self):
        return len(self.segment_id)


    def __getitem__(self, idx):
        samples = {}
        cur_mov = self.segment_id[idx]
        # print(cur_mov)
        # print(cur_mov)
        # print(len(self.segment_id))
        # exit()
        if self.cur_experiment in ["V", "A", "PE"]:
            cur_label = int(self.segment_label[idx] - 1)

        # 音频进行padding
        audio_tensor = torch.zeros((self.config['audio_max_len'], self.config['input_dim_a']), dtype=torch.float32)
        audio_mask = torch.zeros(self.config['audio_max_len'], dtype=torch.long)
        audio_time = 1
        if self.audio_feature.__contains__(cur_mov):
            cur_audio_feature = self.audio_feature[cur_mov]
            audio_time, audio_dim = cur_audio_feature.shape
            audio_tensor[:audio_time, :] = torch.FloatTensor(cur_audio_feature)
            audio_mask[:audio_time] = 1

        # 视觉特征进行padding
        vision_tensor = torch.zeros((self.config['vision_max_len'], self.config['input_dim_v']), dtype=torch.float32)
        vision_mask = torch.zeros(self.config['vision_max_len'], dtype=torch.long)
        vision_time = 1
        if self.vision_feature.__contains__(cur_mov):
            cur_vision_feature = self.vision_feature[cur_mov]
            vision_time, _, vision_dim = cur_vision_feature.shape
            vision_tensor[:vision_time, :] = torch.FloatTensor(cur_vision_feature[:, 0, :])
            vision_mask[:vision_time] = 1 
            cur_vision_feature = self.vision_feature[cur_mov]
            # 它有三个维度

        # 弹幕特征进行padding
        danmu_tensor = torch.zeros((self.config['text_max_len'], self.config['input_dim_t']), dtype=torch.float32)
        danmu_mask = torch.zeros(self.config['text_max_len'], dtype=torch.long)
        danmu_time = 1
        if self.danmu_feature.__contains__(cur_mov):
            cur_danmu_feature = self.danmu_feature[cur_mov]
            danmu_time, danmu_dim = cur_danmu_feature.shape
            danmu_tensor[:danmu_time, :] = torch.FloatTensor(cur_danmu_feature)
            danmu_mask[:danmu_time] = 1
        else:
            danmu_tensor.fill_(1e-4)
            danmu_mask = torch.ones(self.config['text_max_len'], dtype=torch.long)
        # print(cur_mov, audio_tensor.size(), danmu_tensor.size(), audio_time, danmu_time, cur_label)
        # print(cur_mov)
        return {
            'idx':cur_mov,
            'text':danmu_tensor,
            'text_mask':danmu_mask,
            #'text_time':danmu_time,
            'vision':vision_tensor,
            'vision_mask':vision_mask,
            #'vision_time':vision_time,
            'audio':audio_tensor,
            'audio_mask':audio_mask,
            #'audio_time':audio_time,
            'classification_labels':cur_label
        }

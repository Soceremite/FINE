import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataloader.base_dataset import BaseDataset

idx_to_label = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'suprise',
}

labels_en = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels_ch = ['愤怒', '厌恶', '恐惧', '高兴', '平静', '悲伤', '惊奇']

class CHERMADataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, type):
        ''' CHERMA dataset reader
            type in ['train', 'valid', 'test']
        '''
        dtype_dic = {'train':'train',
                     'valid':'dev',
                     'test':'test'}
        self.dataset_name  = opt.dataset_name
        self.dataset_type = dtype_dic[type]

        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        with open(os.path.join(pwd, 'config', f'{self.dataset_name}_config.json'), 'r', encoding='UTF-8') as f:
            self.config = json.load(f)
        self.samples_per_class = self.config[f'samples_per_class_{str(opt.num_classes)}']
        if opt.aligned:
            self.dataset_path = self.config['align_data_path']
        else:
            self.dataset_path = self.config['raw_data_path']
        print("loading from: ", self.dataset_path)

        self.filename_label_list = []

        with open(os.path.join(self.dataset_path, self.dataset_type + '.json')) as f:
            for example in json.load(f):
                a = example['audio_file'].replace('.wav', '')
                v = example['video_file']
                self.filename_label_list.append((a, v, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))
    
    def _get_dataparams(self):
        return  {
            "samples_per_class":self.samples_per_class,
            "min_label":self.config[f'min_label'],
            "max_label":self.config[f'max_label'],
            "input_dim_t":self.config['input_dim_t'],
            "input_dim_v":self.config['input_dim_v'],
            "input_dim_a":self.config['input_dim_a']
        }

    def __len__(self):
        return len(self.filename_label_list)


    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]
        text_path = os.path.join(self.dataset_path, f'text/{self.dataset_type}/{current_filename}.npy')
        text_vector = np.load(text_path)
        text_vector = torch.from_numpy(text_vector)
        # print(text_vector.shape)
        video_path = os.path.join(self.dataset_path, f'visual/{self.dataset_type}/{current_filename}.mp4.npy')
        video_vector = np.load(video_path)
        video_vector = torch.from_numpy(video_vector)
        # print(video_vector.shape)
        audio_path = os.path.join(self.dataset_path, f'audio/{self.dataset_type}/{current_filename}.npy')
        audio_vector = np.load(audio_path)
        audio_vector = torch.from_numpy(audio_vector)
        if audio_vector.shape[0]>self.config['audio_max_len']:
            audio_vector = audio_vector[:self.config['audio_max_len']]
        pad_length = self.config['audio_max_len'] - audio_vector.shape[0]
        audio_lengths = audio_vector.shape[0]
        audio_padding = np.zeros((pad_length, self.config['input_dim_a']))
        audio_vector = np.concatenate((audio_vector, audio_padding))
        return {
            'text':text_vector,
            'text_lengths':text_vector.shape[0],
            'vision':video_vector,
            'vision_lengths':video_vector.shape[0],
            'audio':audio_vector,
            'audio_lengths':audio_lengths,
            'classification_labels':labels_ch.index(label_m),
            'classification_labels_t':labels_ch.index(label_t),
            'classification_labels_a':labels_ch.index(label_a),
            'classification_labels_v':labels_ch.index(label_v)
        }

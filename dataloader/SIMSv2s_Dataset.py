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


class SIMSv2sDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, type):
        ''' MOSEI dataset reader
            type in ['train', 'valid', 'test']
        '''
        self.dataset_name  = opt.dataset_name
        self.dataset_type = type
        self.text_max_length=50

        self.bert_tokenizer = bert_tokenizer(opt.bert_pretrain)
        self.separator = '$_$'
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        with open(os.path.join(pwd, 'config', f'{self.dataset_name}_config.json'), 'r', encoding='UTF-8') as f:
            self.config = json.load(f)
        assert opt.aligned == False , 'CH-SIMS only have raw Dataset'
        data_path = self.config['raw_data_path']
        print(data_path)
        self.samples_per_class = self.config[f'samples_per_class_{str(opt.num_classes)}']
        with open(data_path, 'rb') as f:
            self.data = pk.load(f)[type]

        self.read_keys = ['raw_text', 'audio', 'vision', 'id', 'text', 'audio_lengths', 'vision_lengths', 'regression_labels']

        if 'audio_lengths' not in self.data.keys():
            audio_len = self.data['audio'].shape[1]
            self.data['audio_lengths'] = [audio_len] * self.data['audio'].shape[0]
        if 'vision_lengths' not in self.data.keys():
            vision_len = self.data['vision'].shape[1]
            self.data['vision_lengths'] = [vision_len] * self.data['vision'].shape[0]

        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.data['id'] = np.array(self.data['id'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.index2id = {self.data['id'][i]:i for i in range(len(self.data['id']))}
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
        self.scaled_embedding_averaged = False
        # self.__normalize()
        self.__gen_mask()

    def _get_dataparams(self):

        return  {
            "samples_per_class":self.samples_per_class,
            "min_label":self.config['min_label'],
            "max_label":self.config['max_label'],
            "input_dim_t":self.config['input_dim_t'],
            "input_dim_v":self.config['input_dim_v'],
            "input_dim_a":self.config['input_dim_a']
        }

    def __gen_mask(self):
        vision_tmp = torch.sum(torch.tensor(self.data['vision']), dim=-1)
        vision_mask = (vision_tmp == 0)

        for i in range(self.size):
            vision_mask[i][0] = False
        vision_mask = torch.cat((vision_mask[:, 0:1], vision_mask), dim=-1)

        self.data['vision_padding_mask'] = vision_mask
        audio_tmp = torch.sum(torch.tensor(self.data['audio']), dim=-1)
        audio_mask = (audio_tmp == 0)
        for i in range(self.size):
            audio_mask[i][0] = False
        audio_mask = torch.cat((audio_mask[:, 0:1], audio_mask), dim=-1)
        self.data['audio_padding_mask'] = audio_mask

    def __pad(self):

        PAD = torch.zeros(self.data['vision'].shape[0], 1, self.data['vision'].shape[2])
        self.data['vision'] = np.concatenate((self.data['vision'], PAD), axis=1)
        Ones = torch.ones(self.data['vision'].shape[0], self.data['vision'].shape[2])
        for i in range(len(self.data['vision'])):
            self.data['vision'][i, self.data['vision_lengths'], :] = Ones

        PAD = torch.zeros(self.data['audio'].shape[0], 1, self.data['audio'].shape[2])
        self.data['audio'] = np.concatenate((self.data['audio'], PAD), axis=1)
        Ones = torch.ones(self.data['audio'].shape[0], self.data['audio'].shape[2])
        for i in range(len(self.data['audio'])):
            self.data['audio'][i, self.data['audio_lengths'], :] = Ones

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.data['vision'] = np.mean(self.data['vision'], axis=0, keepdims=True)
        self.data['audio'] = np.mean(self.data['audio'], axis=0, keepdims=True)

        # remove possible NaN values
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0

        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))

    def __scale(self):
        self.scaled_audio = self.data['audio'].copy()
        self.scaled_vision = self.data['vision'].copy()
        self.scaled_text = self.data['text'].copy()
        for i in range(self.audio_fea_size[-1]):
            max_num = np.max(self.data['audio'][:, :, i])
            min_num = np.min(self.data['audio'][:, :, i])
            self.scaled_audio[:, :, i] = (self.data['audio'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        for i in range(self.vision_fea_size[-1]):
            max_num = np.max(self.data['vision'][:, :, i])
            min_num = np.min(self.data['vision'][:, :, i])
            self.scaled_vision[:, :, i] = (self.data['vision'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1

        self.scaled_audio = torch.tensor(self.scaled_audio)
        self.scaled_vision = torch.tensor(self.scaled_vision)
        self.scaled_text = torch.tensor(self.scaled_text)

    def __pre_sample(self, _rank, _label):
        retrieve = {'ss': [],
                    'sd': [],
                    'ds': [],
                    'dd': [],
                    }
        for i in range(self.size):
            _ss = []
            _sd = []
            # _ds = []
            _dd = []
            for j in range(int(self.size / 10)):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    _ss.append(int(_rank[i][j]))
                else:
                    _sd.append(int(_rank[i][j]))
            for j in range(-1, -int(self.size / 10), -1):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    continue
                    # _ds.append(int(_rank[i][j]))
                else:
                    _dd.append(int(_rank[i][j]))
            if len(_ss) < 2 or len(_sd) < 2 or len(_dd) < 2:
                print('Unique sample detected, may cause error!')

            retrieve['ss'].append(_ss[:50])
            retrieve['sd'].append(_sd[:50])
            # retrieve['ds'].append(_ds[:50])
            retrieve['dd'].append(_dd[:50])
        return retrieve

    def update_matrix(self, T, V, A):
        self.scaled_text = T
        self.scaled_vision = V
        self.scaled_audio = A
        self.scaled_embedding_averaged = True
        # self.__gen_cos_matrix()

    def sample(self, _sample_idx):
        samples2 = {}
        idx2 = []
        for i in _sample_idx:
            idx2 += random.sample(self.M_retrieve['ss'][i], 2)
            idx2 += random.sample(self.M_retrieve['dd'][i], 2)
            idx2 += random.sample(self.M_retrieve['sd'][i], 2)
        for key in self.data:
            if type(self.data[key]) == list:
                continue
            else:
                if type(self.data[key][0]) == np.str_:
                    samples2[key] = self.data[key][idx2].tolist()
                else:
                    if type(self.data[key]) is not torch.Tensor:
                        samples2[key] = torch.tensor(self.data[key][idx2])
                    else:
                        samples2[key] = self.data[key][idx2]
        return samples2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.read_keys:
            samples[key] = self.data[key][idx]
        tokenized_text = self.bert_tokenizer(
                self.data['raw_text'][idx], 
                max_length = self.text_max_length,
                padding = "max_length",
                truncation = True,
                add_special_tokens = True,
                return_attention_mask = True
            )
        samples['input_ids'] = torch.tensor(tokenized_text['input_ids'])
        samples['attention_mask'] = torch.tensor(tokenized_text['attention_mask'])

        return samples

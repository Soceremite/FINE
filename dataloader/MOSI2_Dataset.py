import torch
import numpy as np
import random
import os
import json
import _pickle as pk
from torch.utils.data import Dataset, TensorDataset
from dataloader.base_dataset import BaseDataset
from models.networks.modal_encoder import bert_tokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MOSI2Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, type):
        ''' MOSI dataset reader
            type in ['train', 'valid', 'test']
        '''
        self.dataset_name  = opt.dataset_name
        self.dataset_type = type
        self.text_context_length = opt.text_context_length
        self.audio_context_length = opt.audio_context_length
        self.bert_pretrain = opt.bert_pretrain

        self.bert_tokenizer = bert_tokenizer(self.bert_pretrain)
        self.separator = '$_$'
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        with open(os.path.join(pwd, 'config', f'{self.dataset_name}_config.json'), 'r', encoding='UTF-8') as f:
            self.config = json.load(f)
        data_path = self.config['data_path']
        print("loading from: ", data_path)
        self.samples_per_class = self.config[f'samples_per_class_{str(opt.num_classes)}']
        dtype_dic = {'train':'train',
                     'valid':'dev',
                     'test':'test'}
        with open(data_path, 'rb') as f:
            self.data = pk.load(f)[dtype_dic[self.dataset_type]]

        self.size = len(self.data)
    def _get_dataparams(self):

        return  {
            "samples_per_class":self.samples_per_class,
            "min_label":self.config['min_label'],
            "max_label":self.config['max_label'],
            "input_dim_t":self.config['input_dim_t'],
            "input_dim_v":self.config['input_dim_v'],
            "input_dim_a":self.config['input_dim_a']
        }
    
    def prepare_bert_input(self, tokens, visual, acoustic, tokenizer):
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, self.config['input_dim_a']))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
        # acoustic_len = acoustic.shape[0]

        visual_zero = np.zeros((1, self.config['input_dim_v']))
        visual = np.concatenate((visual_zero, visual, visual_zero))
        # visual_len = visual.shape[0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = self.config['max_seq_length'] - len(input_ids)

        acoustic_padding = np.zeros((pad_length, self.config['input_dim_a']))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, self.config['input_dim_v']))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputs
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, visual, acoustic, input_mask, segment_ids

    def __len__(self):
        return self.size

    def __getitem__(self, idx):


        (words, visual, acoustic), regression_labels, segment_id = self.data[idx]

        # print(words)
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = self.bert_tokenizer.tokenize(word)
        # print(tokenized)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > self.config['max_seq_length'] - 2:
            tokens = tokens[: self.config['max_seq_length'] - 2]
            acoustic = acoustic[: self.config['max_seq_length'] - 2]
            visual = visual[: self.config['max_seq_length'] - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = self.prepare_bert_input(
            tokens, visual, acoustic, self.bert_tokenizer
        )

        # Check input length
        assert len(input_ids) == self.config['max_seq_length']
        assert len(input_mask) == self.config['max_seq_length']
        assert len(segment_ids) == self.config['max_seq_length']
        assert acoustic.shape[0] == self.config['max_seq_length']
        assert visual.shape[0] == self.config['max_seq_length']

        input_ids = np.array(input_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        segment_ids = np.array(segment_ids, dtype=np.int64)
        visual = np.array(visual, dtype=np.float32)
        acoustic = np.array(acoustic, dtype=np.float32)
        regression_labels = regression_labels[0][0]   #直接取出数字

        audio_lengths = vision_lengths= sum(input_mask)

        
        # 返回一个字典
        return {
            'input_ids': input_ids,
            'vision': visual,
            'vision_lengths':vision_lengths,
            'audio': acoustic,
            'audio_lengths':audio_lengths,
            'attention_mask': input_mask,
            'segment_ids': segment_ids,
            'regression_labels': regression_labels
        }



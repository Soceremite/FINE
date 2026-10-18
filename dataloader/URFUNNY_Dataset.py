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


class URFUNNYDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, type):
        ''' UN-FUNNY dataset reader
            type in ['train', 'valid', 'test']
        '''
        self.dataset_name  = opt.dataset_name
        self.dataset_type = type
        self.text_context_length = opt.text_context_length
        self.audio_context_length = opt.audio_context_length
        self.bert_pretrain = opt.bert_pretrain

        self.visual_features_list=list(range(55,91))
        self.acoustic_features_list=list(range(0,60))

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
    
    def prepare_bert_input(self, tokens_a, tokens_b, visual_a, visual_b, acoustic_a, acoustic_b, hcf_a, hcf_b, tokenizer):
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, self.config['input_dim_a_origin']))
        if len(tokens_a) == 0:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_zero, acoustic_b, acoustic_zero)
            )
        else:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero)
            )
        # acoustic_len = acoustic.shape[0]

        visual_zero = np.zeros((1, self.config['input_dim_v_origin']))
        if len(tokens_a) == 0:
            visual = np.concatenate((visual_zero, visual_zero, visual_b, visual_zero))
        else:
            visual = np.concatenate(
                (visual_zero, visual_a, visual_zero, visual_b, visual_zero)
            )

        hcf_zero = np.zeros((1,4))
        if len(tokens_a) == 0:
            hcf = np.concatenate((hcf_zero, hcf_zero, hcf_b, hcf_zero))
        else:
            hcf = np.concatenate(
                (hcf_zero, hcf_a, hcf_zero, hcf_b, hcf_zero)
                
            )

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)

        acoustic_padding = np.zeros(
            (self.config['max_seq_length'] - len(input_ids), acoustic.shape[1])
        )
        acoustic = np.concatenate((acoustic, acoustic_padding))
        #original urfunny acoustic feature dimension is 81.
        #we found many features are highly correllated. so we removed
        #highly correlated feature to reduce dimension
        acoustic=np.take(acoustic, self.acoustic_features_list,axis=1)
        
        visual_padding = np.zeros(
            (self.config['max_seq_length'] - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))
        #original urfunny visual feature dimension is more than 300.
        #we only considred the action unit and face shape parameter features
        visual = np.take(visual, self.visual_features_list,axis=1)

        hcf_padding= np.zeros(
            (self.config['max_seq_length'] - len(input_ids), hcf.shape[1])
        )
        
        hcf = np.concatenate((hcf, hcf_padding))


        padding = [0] * (self.config['max_seq_length'] - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, visual, acoustic, hcf, input_mask, segment_ids

    def __len__(self):
        return self.size
    
    def get_inversion(self, tokens, SPIECE_MARKER="_"):
        inversion_index = -1
        inversions = []
        for token in tokens:
            if SPIECE_MARKER in token:
                inversion_index += 1
            inversions.append(inversion_index)
        return inversions
    
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        pop_count = 0
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) == 0:
                tokens_b.pop()
            else:
                pop_count += 1
                tokens_a.pop(0)
        return pop_count

    def __getitem__(self, idx):

        #p denotes punchline, c deontes context
        #hid is the utterance unique id. these id's are provided by the authors of urfunny and mustard
        #label is either 1/0 . 1=humor, 0=not humor
        (
            (p_words, p_visual, p_acoustic, p_hcf),# 
            (c_words, c_visual, c_acoustic, c_hcf),# 
            hid,
            label
        ) = self.data[idx]
        text_a = ". ".join(c_words)
        text_b = p_words + "."
        tokens_a = self.bert_tokenizer.tokenize(text_a)
        tokens_b = self.bert_tokenizer.tokenize(text_b)
        
        inversions_a = self.get_inversion(tokens_a)
        inversions_b = self.get_inversion(tokens_b)

        pop_count = self._truncate_seq_pair(tokens_a, tokens_b, self.config['max_seq_length'] - 3)

        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[: len(tokens_b)]


        visual_a = []
        acoustic_a = []
        hcf_a=[]        
        #our acoustic and visual features are aligned on word level. So we just 
        #create copy of the same visual/acoustic vectors that belong to same word.
        #because ber tokenizer split word into subwords
        for inv_id in inversions_a:
            visual_a.append(c_visual[inv_id, :])
            acoustic_a.append(c_acoustic[inv_id, :])
            hcf_a.append(c_hcf[inv_id, :])
            
        visual_a = np.array(visual_a)
        acoustic_a = np.array(acoustic_a)
        hcf_a = np.array(hcf_a)
        
        visual_b = []
        acoustic_b = []
        hcf_b = []
        for inv_id in inversions_b:
            visual_b.append(p_visual[inv_id, :])
            acoustic_b.append(p_acoustic[inv_id, :])
            hcf_b.append(p_hcf[inv_id, :])
        
        visual_b = np.array(visual_b)
        acoustic_b = np.array(acoustic_b)
        hcf_b = np.array(hcf_b)

        input_ids, visual, acoustic, hcf, input_mask, segment_ids = self.prepare_bert_input(
            tokens_a, tokens_b, visual_a, visual_b, acoustic_a, acoustic_b, hcf_a, hcf_b, self.bert_tokenizer
        )

        # Check input length
        assert len(input_ids) == self.config['max_seq_length']
        assert len(input_mask) == self.config['max_seq_length']
        assert len(segment_ids) == self.config['max_seq_length']
        assert acoustic.shape[0] == self.config['max_seq_length']
        assert visual.shape[0] == self.config['max_seq_length']
        assert hcf.shape[0] == self.config['max_seq_length']
        label_id = float(label)

        input_ids = np.array(input_ids, dtype=np.int64)
        input_mask = np.array(input_mask, dtype=np.int64)
        segment_ids = np.array(segment_ids, dtype=np.int64)
        visual = np.array(visual, dtype=np.float32)
        acoustic = np.array(acoustic, dtype=np.float32)
        hcf = np.array(hcf, dtype=np.float32)
        label_id = label_id   #直接取出数字

        audio_lengths = vision_lengths= sum(input_mask)

        return {
            'input_ids': input_ids,
            'vision': visual,
            'vision_lengths':vision_lengths,
            'audio': acoustic,
            'audio_lengths':audio_lengths,
            'attention_mask': input_mask,
            'segment_ids': segment_ids,
            'classification_labels': label,
            'hcf':hcf,
            'label_id':label_id
        }



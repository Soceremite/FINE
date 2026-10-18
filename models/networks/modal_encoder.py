import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def bert_tokenizer(config_path='bert-base-uncased'):
    return AutoTokenizer.from_pretrained(config_path)

class BertEncoder(nn.Module):
    def __init__(self, finetune=True, pretrain='bert-base-uncased'):
        super().__init__()
        # self.config = BertConfig.from_pretrained(pretrain)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        self.encoder = AutoModel.from_pretrained(pretrain)
        self.finetune=finetune
        self.layer_ids = [-4, -3, -2, -1]
    
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text=None, input_ids=None, attention_mask=None, device=None):
        if text!=None:
            if device is None:
                device = self.encoder.device
            
            tokenized_text = self.tokenizer(
                                        text,            
                                        max_length = 96,                                
                                        padding = "max_length",     # Pad to the specified max_length. 
                                        truncation = True,          # Truncate to the specified max_length. 
                                        add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                                        return_attention_mask = True            
            )
            input_ids, attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']
            input_ids_tensor = torch.tensor(input_ids).to(device)
            attention_mask_tensor = torch.tensor(attention_mask).to(device)
        else:
            assert input_ids!=None,'text is None, input_ids  must be not None'
            assert attention_mask!=None,'text is None, attention_mask  must be not None'
            input_ids_tensor = input_ids
            attention_mask_tensor = attention_mask

        

        if self.finetune:
            sequence_output = self.encoder(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor).last_hidden_state
        else:
            with torch.no_grad():
                sequence_output = self.encoder(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor).last_hidden_state
        return sequence_output
    

class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, fea_size, tf_hidden_dim, drop_out=0.5):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        num_patches = 500
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):

    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward, 
                 num_layers, 
                 dropout=0.2, 
                 activation='gelu'):
        super(TfEncoder, self).__init__()
        self.model_type = 'vision_encoder'
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining()

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)

class VisionEncoder(nn.Module):
    def __init__(self, 
                 fea_size, 
                 encoder_fea_dim, 
                 nhead, 
                 dim_feedforward,
                 num_layers,
                 drop_out=0.5):
        super(VisionEncoder, self).__init__()

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,
                                 dropout=drop_out, activation='gelu')
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)
        # self.fc = nn.Linear(709,768)

    def forward(self, vision, key_padding_mask):
        x = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        # x = torch.mean(x, dim=-2, keepdim=True)
        return x
    

class AudioEncoder(nn.Module):
    def __init__(self, 
                 fea_size, 
                 encoder_fea_dim, 
                 nhead, 
                 dim_feedforward,
                 num_layers,
                 drop_out=0.5):
        super(AudioEncoder, self).__init__()

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,
                                 dropout=drop_out, activation='gelu')
        self.layernorm = nn.LayerNorm(encoder_fea_dim)

    def forward(self, audio, key_padding_mask):
        x = self.encoder(audio, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        # x = torch.mean(x, dim=-2, keepdim=True)

        return x
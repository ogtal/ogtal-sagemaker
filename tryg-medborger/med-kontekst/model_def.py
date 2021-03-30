from transformers import ElectraModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class ElectraClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(self.electra.config.hidden_size+20, self.electra.config.hidden_size+20)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size+20, self.num_labels)

    def classifier(self,x):
        # x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids=None,attention_mask=None,group_feat=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        sequence_output = sequence_output[:, 0, :]
        features = torch.cat((sequence_output,group_feat.squeeze()),dim=1)

        logits = self.classifier(features)
        return logits
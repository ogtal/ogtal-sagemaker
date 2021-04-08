from transformers import ElectraModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class ElectraClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(pretrained_model_name)

        self.lstm = nn.LSTM(256, 512, batch_first=True,bidirectional=True)
        self.dense = nn.Linear(1024, 1024)
        self.dense1 = nn.Linear(1024, self.num_labels)

        # self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        # self.dense = nn.Linear(512, self.num_labels)

    def classifier(self,sequence_output):

        x,(h,c) = self.lstm(sequence_output)
        x = F.gelu(self.dense(x[:,-1]))
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense1(x))
        x = self.dropout(x)
    
        return x


        # raise ValueError()

        # x = sequence_output[:, 0, :]
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # x = F.gelu(self.dense(x))
        # x = self.dropout(x)
        # logits = self.out_proj(x)
        # return logits

    def forward(self, input_ids=None,attention_mask=None):
        discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits
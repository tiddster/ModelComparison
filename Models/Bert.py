import torch.nn as nn
from transformers import BertForSequenceClassification
import torch.nn.functional as F


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.Bert = BertForSequenceClassification.from_pretrained(config.bert_path, num_labels=config.n_class)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.Bert(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        return loss, F.softmax(logits)
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(config.vocab_size+1, config.embedding_dim)
        self.Bi_LSTM = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.n_class)

    def forward(self, inputs):
        inputs = self.embeddings(inputs)
        inputs = inputs.transpose(0, 1)

        outputs, (hidden_state, cell_state) = self.Bi_LSTM(inputs)

        outputs = outputs[-1]

        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = F.softmax(outputs)
        return outputs

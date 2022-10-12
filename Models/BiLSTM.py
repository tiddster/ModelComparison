import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(config.vocab_size+1, config.embedding_dim)
        self.bilstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, bidirectional=True)
        self.fc = nn.Linear(100 * config.hidden_size * 2, config.n_class)

    def forward(self, X):
        inputs = self.embeddings(X)
        inputs = inputs.transpose(0,1)

        outputs, (_, _) = self.bilstm(inputs)
        outputs = outputs.transpose(0,1)

        outputs = outputs.reshape((outputs.shape[0], -1))
        outputs = self.fc(outputs)

        return F.softmax(outputs)
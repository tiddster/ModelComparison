import torch


class Config():
    def __init__(self, vocab_size, n_class):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = 300
        self.hidden_size = 300
        self.lr = 0.01
        self.dropout = 0.01

        self.device = torch.device('cpu')
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_optimizer(self, net):
        return torch.optim.Adam(net.parameters(), lr=self.lr)

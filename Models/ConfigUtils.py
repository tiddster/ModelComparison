import torch


class Config():
    def __init__(self, vocab_size, n_class, hidden_size=300, embedding_dim=300):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lr = 0.002
        self.dropout = 0.01

        self.device = torch.cuda.set_device(0)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.bert_path = ""

    def get_optimizer(self, net):
        return torch.optim.SGD(net.parameters(), lr=self.lr)

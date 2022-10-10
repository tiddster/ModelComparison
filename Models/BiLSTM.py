import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(config.vocab_size + 1, config.embedding_dim)
        self.Bi_LSTM = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.n_class)

    def Attenion(self, lstm_output, final_state):
        # lstm_output : [batch_size, seq_len, num_hidden * num_directions(=2)], F matrix
        # final_state : [num_directions(=2), batch_size, num_hidden]
        batch_size = len(lstm_output)
        # hidden=[batch_size, num_hidden*num_directions(=2), 1]
        hidden = final_state.view((batch_size, -1, 1))

        # torch.bmm为多维矩阵的乘法：a=[b, h, w], c=[b,w,m]  bmm(a,b)=[b,h,m], 也就是对每一个batch都做矩阵乘法
        # squeeze(2), 判断第三维上维度是否为1，若为1则去掉
        # attn_weights:
        # = [batch_size, seq_len, num_hidden * num_directions(=2)] @  [batch_size, num_hidden*num_directions(=2), 1]
        # = [batch_size, seq_len, 1]
        attn_weights = lstm_output @ hidden

        soft_attn_weights = F.softmax(attn_weights, 1)

        # context
        # = [batch_size, num_hidden * num_directions(=2), seq_len] @  [batch_size, seq_len, 1]
        # = [batch_size, num_hidden * num_directions]
        context = (lstm_output.transpose(1, 2) @ soft_attn_weights).squeeze(2)

        return context, soft_attn_weights

    def forward(self, X):
        """
        :param X:[batch_size, seq_len]
        :return:
        """
        # inputs: [batch_size, seq_len, embedding_dim]
        inputs = self.embeddings(X)
        # inputs: [seq_len, batch_size, embedding_dim]
        inputs = inputs.transpose(0,1)

        output, (final_hidden_state, final_cell_state) = self.Bi_LSTM(inputs)
        # output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        # final_hidden_state : [num_directions, batch_size, num_hidden]
        output = output.transpose(0, 1)
        attn_output, attention = self.Attenion(output, final_hidden_state)

        # attn_output : [batch_size, num_classes], attention : [batch_size, seq_len, 1]
        return self.fc(attn_output)
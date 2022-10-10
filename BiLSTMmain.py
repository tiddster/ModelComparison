import time

import torch

import LoadDatasets.SST as sst
from Models.ConfigUtils import Config
from Models.BiLSTM import BiLSTM

def train(net, train_iter, test_iter):
    for epoch in range(1000):
        train_total, train_acc_total, total_loss, start, batch_num = 0, 0, 0.0, time.time(), 0
        for x, y in train_iter:
            x, y = x.to(config.device), y.to(config.device)
            outputs = net(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

            train_total += y.shape[0]
            train_acc_total += (outputs.argmax(dim=1) == y).sum().item()
            # print(outputs.argmax(dim=1), y)

        if (epoch + 1) % 2 == 0:
            train_acc = train_acc_total / train_total

            test_total, test_acc_total = 0, 0

            for test_x, test_y in test_iter:
                test_x, test_y = test_x.to(config.device), test_y.to(config.device)
                test_outputs = net(test_x)

                test_total += test_y.shape[0]
                test_acc_total += (test_outputs.argmax(dim=1) == test_y).sum().item()

            test_acc = test_acc_total / test_total
            end = time.time()
            print(f"[EPOCH]: {epoch}, ", "train_acc: {:.4f}%".format(train_acc * 100),
                  "test_acc: {:.4f}%".format(test_acc * 100), f"avg_loss: {total_loss/batch_num}", f"time: {end-start}")


if __name__ == '__main__':
    print(torch.cuda.device_count())

    train_iter, test_iter, _, vocab_size = sst.get_data_info(32)

    print(f"进行数据集设置, 词向量大小为：{vocab_size}")
    config = Config(vocab_size, 3)

    net = BiLSTM(config).to(config.device)
    criterion = config.criterion
    optimizer = config.get_optimizer(net)

    train(net, train_iter, test_iter)

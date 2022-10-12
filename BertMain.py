import time

import torch

import LoadDatasets.SST as SST
from Models.ConfigUtils import Config
from Models.Bert import BertModel

def train():
    for epoch in range(12):
        train_total, train_acc_total, total_loss, start, batch_num = 0, 0, 0.0, time.time(), 0
        for batch in train_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            train_labels = batch['label'].to(device)
            loss, train_outputs = net(input_ids, attention_mask=attention_mask, labels=train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

            train_total += train_labels.shape[0]
            train_acc_total += (train_outputs.argmax(dim=1) == train_labels).sum().item()
            # print(outputs.argmax(dim=1), y)
        if (epoch + 1) % 1 == 0:
            train_acc = train_acc_total / train_total

            test_total, test_acc_total = 0, 0

            for batch in test_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                test_labels = batch['label'].to(device)
                loss, test_outputs = net(input_ids, attention_mask=attention_mask, labels=test_labels)

                test_total += test_labels.shape[0]
                test_acc_total += (test_outputs.argmax(dim=1) == test_labels).sum().item()

            test_acc = test_acc_total / test_total
            end = time.time()
            print(f"[EPOCH]: {epoch}, ", "train_acc: {:.4f}%".format(train_acc * 100),
                  "test_acc: {:.4f}%".format(test_acc * 100), f"avg_loss: {total_loss/batch_num}", f"time: {end-start}")


if __name__ == "__main__":
    train_iter, test_iter = SST.get_bert_data_info(16)

    config = Config(0, 5, hidden_size=768)

    device = config.device

    config.bert_path = SST.bert_path
    net = BertModel(config).to(device)

    criterion = config.criterion
    optimizer = config.get_optimizer(net)

    train()
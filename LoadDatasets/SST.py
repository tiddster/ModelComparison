import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

"""
BiLSTM + Attention   5分类：35%    3分类   49%
"""

root_path = "D:\新建文件夹\Dataset\SST\\"
bert_path = "D:\新建文件夹\Dataset\Bert-uncased"

tokenizer = BertTokenizer.from_pretrained(bert_path)
max_len = 100


def to_npz(fileName):
    txt_file = root_path + fileName + ".csv"
    npz_file = root_path + fileName + ".npz"

    data = pd.read_csv(txt_file)
    inputs = list(data["INPUT"])
    outputs = list(data["OUTPUT"])

    labelList = []
    for l in outputs:
        if 1 <= l < 3:
            labelList.append(0)
        elif l == 3:
            labelList.append(1)
        else:
            labelList.append(2)

    contextList = []
    max_id = 0
    for text in inputs:
        token = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(token)
        max_id = max(max_id, max(ids))
        contextList.append(ids)

    for i in range(len(contextList)):
        contextList[i] += [0] * (max_len - len(contextList[i]))

    contextList = np.asarray(contextList)
    labelList = np.asarray(labelList)
    max_id = np.asarray(max_id)
    np.savez(npz_file, contextList=contextList, labelList=labelList, max_id=max_id)

    return inputs


def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas


# all_context = np.vstack((train_data["contextList"], test_data["contextList"], val_data["contextList"]))
# scaler = MinMaxScaler(feature_range=[0,10])
# all_context = scaler.fit_transform(all_context)
#
# train_context = all_context[:len(train_data["contextList"])]
# train_labels = train_data["labelList"]
# test_context = all_context[len(train_data["contextList"]):len(train_data["contextList"])+len(test_data["contextList"])]
# test_labels = test_data["labelList"]
# val_context = all_context[len(train_data["contextList"])+len(test_data["contextList"]):]
# val_labels = val_data["labelList"]

class SSTDataset(Dataset):
    def __init__(self, data):
        self.contextList = torch.from_numpy(data["contextList"]).long()
        self.labelList = torch.from_numpy(data["labelList"]).long()

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, index):
        return self.contextList[index], self.labelList[index]


def get_data_info(batch_size):
    # to_npz("train")
    # to_npz("test")

    train_data = read_npz("train")
    test_data = read_npz("test")

    train_dataset = SSTDataset(train_data)
    test_dataset = SSTDataset(test_data)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter, int(max(train_data["max_id"], test_data["max_id"]))



# to_npz("val")

class BertSSTDataset(Dataset):
    def __init__(self, encodings, labels):
        super(BertSSTDataset, self).__init__()
        self.encodings = encodings
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['label'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

def get_bert_data_info(batch_size):
    train_contexts = to_npz("train")
    test_contexts = to_npz("test")

    train_data = read_npz("train")
    test_data = read_npz("test")

    train_labels = train_data["labelList"]
    test_labels = test_data["labelList"]

    train_encoding = tokenizer(train_contexts, truncation=True, padding=True, max_length=100)
    test_encoding = tokenizer(test_contexts, truncation=True, padding=True, max_length=100)

    train_dataset = BertSSTDataset(train_encoding, train_labels)
    test_dataset = BertSSTDataset(test_encoding, test_labels)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter
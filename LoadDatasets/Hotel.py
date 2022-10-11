import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

"""
Hotel数据集： 50 epoch on CUDA
"""

root_path = "D:\新建文件夹\Dataset\HotalBySongbo\\"
bert_path = "D:\新建文件夹\Dataset\Bert-base-Chinese"

tokenizer = BertTokenizer.from_pretrained(bert_path)


def to_npz(fileName):
    txt_file = root_path + fileName + ".txt"
    npz_file = root_path + fileName + ".npz"

    labelList = []
    contextList = []
    max_len = 100
    max_id = 0
    contexts = []

    with open(txt_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            label, context = l.split("    ")

            context = context.replace("\n", "")

            token = tokenizer.tokenize(context)
            if len(token) != 0 and len(token) < max_len:
                ids = tokenizer.convert_tokens_to_ids(token)
                max_id = max(max_id, max(ids))
                contextList.append(ids)
                if label == "-1":
                    label = "0"
                labelList.append(int(label))
                contexts.append(context)

    maskList = []
    for i in range(len(contextList)):
        maskList = [1] * len(contextList[i]) + [0] * (max_len - len(contextList[i]))
        contextList[i] += [0] * (max_len - len(contextList[i]))

    contextList = np.asarray(contextList)
    labelList = np.asarray(labelList)
    maskList = np.asarray(maskList)
    np.savez(npz_file, contextList=contextList, labelList=labelList, maskList=maskList, max_id=max_id)

    return contexts

def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas


class HotelDataset(Dataset):
    def __init__(self, contextList, labelList):
        self.contextList = torch.from_numpy(contextList).long()
        self.labelList = torch.from_numpy(labelList).long()
        print(self.labelList)

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, index):
        return self.contextList[index], self.labelList[index]


def get_data_info(batch_size):
    # to_npz("all")
    data = read_npz("all")

    all_labels = data["labelList"]
    all_context = data["contextList"]

    train_context, test_context, train_labels, test_labels = train_test_split(all_context, all_labels, test_size=0.3,
                                                                              random_state=1)

    train_dataset = HotelDataset(train_context, train_labels)
    test_dataset = HotelDataset(test_context, test_labels)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter, int(data["max_id"])


class BertHotelDataset(Dataset):
    def __init__(self, encodings, labels):
        super(BertHotelDataset, self).__init__()
        self.encodings = encodings
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['label'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

def get_bert_data_info(batch_size):
    contexts = to_npz("all")

    all_data = read_npz("all")
    all_labels = all_data["labelList"]

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_contexts, test_contexts, train_labels, test_labels = train_test_split(contexts, all_labels, test_size=0.3, random_state=1)

    train_encoding = tokenizer(train_contexts, truncation=True, padding=True, max_length=100)
    test_encoding = tokenizer(test_contexts, truncation=True, padding=True, max_length=100)

    train_dataset = BertHotelDataset(train_encoding, train_labels)
    test_dataset = BertHotelDataset(test_encoding, test_labels)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter
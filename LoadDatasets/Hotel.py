import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import  BertTokenizer
from torch.utils.data import Dataset, DataLoader

root_path = "F:\Dataset\HotalBySongbo\\"
bert_vocab = "F:\Dataset\Bert-base-Chinese"

tokenizer = BertTokenizer.from_pretrained(bert_vocab)

def to_npz(fileName):
    txt_file = root_path + fileName + ".txt"
    npz_file = root_path + fileName + ".npz"

    labelList = []
    contextList = []
    max_len = 0

    with open(txt_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            label, context = l.split("    ")

            labelList.append(int(label))

            context = context.replace("\n", "")
            token = tokenizer.tokenize(context)
            ids = tokenizer.convert_tokens_to_ids(token)
            contextList.append(ids)

            max_len = max(max_len, len(ids))

    for i in range(len(contextList)):
        contextList[i] += [0] * (max_len - len(contextList[i]))

    contextList = np.asarray(contextList)
    labelList = np.asarray(labelList)
    np.savez(npz_file, contextList=contextList, labelList=labelList)


def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas


to_npz("all")
data = read_npz("all")

all_labels = data["labelList"]
all_context = data["contextList"]
scaler = MinMaxScaler(feature_range=[0,10])
all_context = scaler.fit_transform(all_context)

train_context, test_context,  train_labels, test_labels = train_test_split(all_context, all_labels, test_size=0.3, random_state=1)


class HotelDataset(Dataset):
    def __init__(self, contextList, labelList):
        self.contextList = torch.from_numpy(contextList).float()
        self.labelList = torch.from_numpy(labelList).float()
        print(self.contextList)

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, index):
        return self.contextList(index), self.labelList(index)


train_dataset = HotelDataset(train_context, train_labels)
test_dataset = HotelDataset(test_context, test_labels)


def get_iter(batch_size):
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_iter, test_iter

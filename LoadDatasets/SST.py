import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

root_path = "F:\Dataset\SST\\"
bert_vocab = "F:\Dataset\Bert-uncased\\vocab.txt"

tokenizer = BertTokenizer.from_pretrained(bert_vocab)
max_len = 100

def to_npz(fileName):
    txt_file = root_path + fileName + ".csv"
    npz_file = root_path + fileName + ".npz"

    data = pd.read_csv(txt_file)
    inputs = list(data["INPUT"])
    outputs = list(data["OUTPUT"])

    contextList = []
    for text in inputs:
        token = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(token)
        contextList.append(ids)

    for i in range(len(contextList)):
        contextList[i] += [0] * (max_len - len(contextList[i]))
        contextList[i] = np.asarray(contextList[i])

    contextList = np.asarray(contextList)
    outputs = np.asarray(outputs)
    np.savez(npz_file, contextList=contextList, labelList=outputs, max_len=max_len)


def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas


# to_npz("train")
# to_npz("test")
# to_npz("val")
train_data = read_npz("train")
test_data = read_npz("test")
val_data = read_npz("val")

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


train_dataset = SSTDataset(train_data)
test_dataset = SSTDataset(test_data)
val_dataset = SSTDataset(val_data)


def get_iter(batch_size):
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_iter, test_iter, val_iter

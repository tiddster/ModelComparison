import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

root_path = "F:\Dataset\\NLPCC2014_sentiment-master\\dataset\\"
bert_path = "F:\Dataset\Bert-base-Chinese"

tokenizers = BertTokenizer.from_pretrained(bert_path)
max_len = 1010

def to_npz(fileName):
    txt_file = root_path + fileName + '.txt'
    save_file = root_path + fileName + '.npz'

    lines = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].replace("\n", "").replace("</review>","")
            if data[i] == '':
                continue
            lines.append(data[i])

    labelList = []
    contexts = []
    if fileName == "train":
        for i in range(len(lines)):
            if "review" in lines[i]:
                continue
            contexts.append(lines[i])
        labelFile = root_path + 'train_labels.txt'
        labelLines = open(labelFile, 'r').readlines()
        for i in range(len(labelLines)):
            labelList.append(int(labelLines[i].replace("\n", "")))
    else:
        for i in range(len(lines)):
            if "\"1\"" in lines[i]:
                labelList.append(1)
            elif "\"0\"" in lines[i]:
                labelList.append(0)
            else:
                contexts.append(lines[i])

    contextList = []
    for context in contexts:
        tokens = tokenizers.tokenize(context)
        ids = tokenizers.convert_tokens_to_ids(tokens)
        contextList.append(ids)

    for i in range(len(contextList)):
        contextList[i] += [0] * (max_len - len(contextList[i]))
        contextList[i] = np.asarray(contextList[i])

    contextList = np.asarray(contextList)
    labelList = np.asarray(labelList)
    np.savez(save_file, contextList=contextList, labelList=labelList)

def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas

# to_npz("train")
# to_npz("test")
train_data = read_npz("train")
test_data = read_npz("test")

all_context = np.vstack((train_data["contextList"], test_data["contextList"]))
scaler = MinMaxScaler(feature_range=[0,10])
all_context = scaler.fit_transform(all_context)

train_context = all_context[:len(train_data["contextList"])]
test_context = all_context[len(train_data["contextList"]):]
train_labels = train_data["labelList"]
test_labels = test_data["labelList"]


class NLPccDataset(Dataset):
    def __init__(self, contextList, labelList):
        self.contextList = torch.from_numpy(contextList).float()
        self.labelList = torch.from_numpy(labelList).float()

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, index):
        return self.contextList(index), self.labelList(index)


train_dataset = NLPccDataset(train_context, train_labels)
test_dataset = NLPccDataset(test_context, test_labels)


def get_iter(batch_size):
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_iter, test_iter

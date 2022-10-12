import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

"""
BiLSTM+Attention   500 epoch on CUDA  
"""

root_path = "D:\新建文件夹\Dataset\\NLPCC2014_sentiment-master\dataset\\"
bert_path = "D:\新建文件夹\Dataset\Bert-base-Chinese"

tokenizers = BertTokenizer.from_pretrained(bert_path)
max_len = 100


def to_npz(fileName):
    txt_file = root_path + fileName + '.txt'
    save_file = root_path + fileName + '.npz'

    lines = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].replace("\n", "").replace("</review>", "")
            if data[i] == '':
                continue
            lines.append(data[i])

    labelList = []
    contexts = []
    max_id = 0
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
        if len(tokens) != 0:
            ids = tokenizers.convert_tokens_to_ids(tokens)
            max_id = max(max_id, max(ids))
            contextList.append(ids)

    for i in range(len(contextList)):
        if len(contextList[i]) < max_len:
            contextList[i] += [0] * (max_len - len(contextList[i]))
        else:
            contextList[i] = contextList[i][:max_len]
        contextList[i] = np.asarray(contextList[i])

    contextList = np.asarray(contextList)
    labelList = np.asarray(labelList)
    max_id = np.asarray(max_id)
    np.savez(save_file, contextList=contextList, labelList=labelList, max_id=max_id)
    return contexts


def read_npz(fileName):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    datas = np.load(root_path + fileName + '.npz')
    np.load.__defaults__ = (None, False, True, 'ASCII')
    print(datas.files)
    return datas


# to_npz("train")
# to_npz("test")

# all_context = np.vstack((train_data["contextList"], test_data["contextList"]))
# scaler = MinMaxScaler(feature_range=[0,10])
# all_context = scaler.fit_transform(all_context)
#
# train_context = all_context[:len(train_data["contextList"])]
# test_context = all_context[len(train_data["contextList"]):]
# train_labels = train_data["labelList"]
# test_labels = test_data["labelList"]


class NLPccDataset(Dataset):
    def __init__(self, data):
        self.contextList = torch.from_numpy(data["contextList"]).long()
        self.labelList = torch.from_numpy(data["labelList"]).long()

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, index):
        return self.contextList[index], self.labelList[index]


def get_data_info(batch_size):
    train_data = read_npz("train")
    test_data = read_npz("test")

    train_dataset = NLPccDataset(train_data)
    test_dataset = NLPccDataset(test_data)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_iter, test_iter, int(max(train_data["max_id"], test_data["max_id"]))


class BertNLPDataset(Dataset):
    def __init__(self, encodings, labels):
        super(BertNLPDataset, self).__init__()
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

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_encoding = tokenizer(train_contexts, truncation=True, padding=True, max_length=100)
    test_encoding = tokenizer(test_contexts, truncation=True, padding=True, max_length=100)

    train_dataset = BertNLPDataset(train_encoding, train_data["labelList"])
    test_dataset = BertNLPDataset(test_encoding, test_data["labelList"])

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter


import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler

"""
该数据集已经存在了预处理数据，可以直接调用
"""
root_path = "F:\Dataset\SemEval 2014\data\\restaurant\\"


def get_npz(fileName):
    data = np.load(root_path + fileName)
    print(data.files)

    return data


class SemEvalLaptop(Dataset):
    def __init__(self, data):
        self.aspectList = torch.from_numpy(data['aspects']).long()
        self.contextList = torch.from_numpy(data['contexts']).long()
        self.labelList = torch.from_numpy(data['labels']).long()
        self.aspect_lens = torch.from_numpy(data['aspect_lens']).long()
        self.context_lens = torch.from_numpy(data['aspect_lens']).long()

        aspect_max_len = self.aspectList.size(1)
        context_max_len = self.contextList.size(1)
        self.aspect_mask = torch.zeros(aspect_max_len, aspect_max_len)
        self.context_mask = torch.zeros(context_max_len, context_max_len)
        for i in range(aspect_max_len):
            self.aspect_mask[i, 0:i + 1] = 1
        for i in range(context_max_len):
            self.context_mask[i, 0:i + 1] = 1

    def __getitem__(self, index):
        return self.aspectList[index], self.contextList[index], self.labelList[index], \
               self.aspect_mask[self.aspect_lens[index] - 1], self.context_mask[self.context_lens[index] - 1]

    def __len__(self):
        return self.labelList.shape[0]


def get_iter():
    train_data = get_npz("train.npz")
    test_data = get_npz("train.npz")
    train_dataset = SemEvalLaptop(train_data)
    test_dataset = SemEvalLaptop(test_data)
    train_iter = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)
    return train_iter, test_iter
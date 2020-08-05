import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class DatasetMetaQA(Dataset):
    def __init__(self, data, word2ix, relations, entities, entity2idx):
        self.data = data
        self.relations = relations
        self.entities = entities
        self.word_to_ix = {}
        self.entity2idx = entity2idx
        self.word_to_ix = word2ix
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())


    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_ids = [self.word_to_ix[word] for word in question_text.split()]
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        return question_ids, head_id, tail_onehot 




def _collate_fn(batch):
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
    longest_sample = sorted_seq_lengths[0]
    minibatch_size = len(batch)
    # print(minibatch_size)
    # aditay
    input_lengths = []
    p_head = []
    p_tail = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    for x in range(minibatch_size):
        # data_a = x[0]
        sample = sorted_seq[x][0]
        p_head.append(sorted_seq[x][1])
        tail_onehot = sorted_seq[x][2]
        p_tail.append(tail_onehot)
        seq_len = len(sample)
        input_lengths.append(seq_len)
        sample = torch.tensor(sample, dtype=torch.long)
        sample = sample.view(sample.shape[0])
        inputs[x].narrow(0,0,seq_len).copy_(sample)

    return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(p_head, dtype=torch.long), torch.stack(p_tail)

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

    


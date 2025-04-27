import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six
import time


class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first:
        padding_value:
        max_len :
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def cache(func):
    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"cache {data_path} doesn't exist")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"cache {data_path} exists")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


def process_cache(unique_key=None):
    """
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "['top_k', 'cut_words', 'max_sen_len']")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            obj = args[0]
            file_path = kwargs['file_path']
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            if not os.path.exists(cache_path):
                data = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                with open(cache_path, 'rb') as f:
                    data = torch.load(f)
            end_time = time.time()
            return data

        return wrapper

    return decorating_function


class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    @process_cache(unique_key=["max_sen_len"])
    def data_process(self, file_path=None):
        raw_iter = open(file_path, encoding="utf8").readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        test_data, _ = self.data_process(file_path=test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(file_path=train_file_path)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(file_path=val_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label









import json
import os
import torch
import random


def load_dataset(dataset):
    def load(path):
        contents = []
        with open(path, 'r') as f:
            jsons = json.load(f, strict=False)
            for j in jsons:
                # strip(): 去除头尾的 \n \r \t ' '(换行、回车、制表符、空格)
                chq = j['chq'].strip()
                faq = j['faq'].strip()
                contents.append((chq, faq, j['perturbations']))
        return contents

    data = load(os.path.join(dataset, 'train_perturbation.json'))

    return data


class DatasetIterator(object):
    def __init__(self, tokenizer, dataset, batch_size, device, contrast_number, max_src_length):
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
        self.tokenizer = tokenizer
        self.n_batches = (len(dataset) - 1) // batch_size
        self.res = False
        if len(dataset) % self.n_batches != 0:
            self.res = True
        self.index = 0
        self.contrast_number = contrast_number
        self.max_src_length = max_src_length

    def text_to_tensor(self, texts):

        o = self.tokenizer.batch_encode_plus(texts, max_length=self.max_src_length, padding='longest', truncation=True,
                                             return_tensors='pt')
        return o['input_ids'].to(self.device, dtype=torch.long), o['attention_mask'].to(self.device, dtype=torch.long)

    def _to_tensor(self, batch):
        src_ids, src_msks = self.text_to_tensor([p[0] for p in batch])
        tgt = self.tokenizer.batch_encode_plus([p[1] for p in batch], max_length=self.max_src_length, padding='longest',
                                               truncation=True, return_tensors='pt')
        y = tgt['input_ids']
        y_ids = y[:, :-1]
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
        tgt_ids = y_ids.to(self.device, dtype=torch.long)
        tgt_msks = tgt['attention_mask'][:, :-1].to(self.device, dtype=torch.long)
        labels = lm_labels.to(self.device, dtype=torch.long)

        texts = []
        for p in batch:
            choice = random.choices(range(len(p[2])), k=self.contrast_number)
            for c in choice:
                texts.append(p[2][c])

        con_ids, con_msks = self.text_to_tensor(texts)
        return src_ids, tgt_ids, con_ids, src_msks, tgt_msks, con_msks, labels

    def __next__(self):
        if self.res and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size:len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def shuffle_data(self):
        random.shuffle(self.dataset)
        self.index = 0

    def __iter__(self):
        return self

    def __len__(self):
        if self.res:
            return self.n_batches + 1
        else:
            return self.n_batches

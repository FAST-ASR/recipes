# Copyright (c) Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
from collections import OrderedDict
import json

from io import BytesIO
import librosa
from subprocess import run, PIPE
import torchaudio

import torch
import kaldi_io
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path


def _collate_fn_train(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    num_spkrs = len(batch[0][3])
    graph_list = [[] for _ in range(num_spkrs)] 
    for i in range(minibatch_size):
        feat, length, utt_id, graphs = batch[i]
        feats[i, :length, :].copy_(feat)
        utt_ids.append(utt_id)
        feat_lengths[i] = length
        for spkr in range(num_spkrs):
            graph_list[spkr].append(graphs[spkr])

    if num_spkrs <= 1:
        graph_list = graph_list[0]

    return feats, feat_lengths, utt_ids, graph_list


def _collate_fn_test(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    for i in range(minibatch_size):
        feat, length, utt_id = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        utt_ids.append(utt_id)
    return feats, feat_lengths, utt_ids


class GraphDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for ChainDatasets.
        """
        super(GraphDataLoader, self).__init__(*args, **kwargs)
        if self.dataset.train:
            self.collate_fn = _collate_fn_train
        else:
            self.collate_fn = _collate_fn_test


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        codes from deepspeech.pytorch 
        https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        """
        super(BucketSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class GraphDataset(data.Dataset):

    def __init__(self, data_json_path, train=True, cache_graph=True, sort=True, no_feat=False):
        super(GraphDataset, self).__init__()
        self.train = train
        self.cache_graph = cache_graph
        self.sort = sort
        self.no_feat = no_feat
        self.samples = []  # list of dicts

        with open(data_json_path, 'rb') as f:
            loaded_json = json.load(f, object_pairs_hook=OrderedDict)

        print("Initializing dataset...")
        for utt_id, val in tqdm(loaded_json.items()):
            sample = {}
            sample['utt_id'] = utt_id
            sample['wav'] = val['wav']
            sample['text'] = val['text']
            sample['duration'] = float(val['duration'])
            if not self.no_feat:
                sample['feat'] = val['feat']
                sample['length'] = int(val['length'])

            if self.train:  # only training data has fst (graph)
                fst_rxfs = val['numerator_fst']
                if isinstance(fst_rxfs, str):
                    fst_rxfs = [fst_rxfs]
                
                sample['graph'] = fst_rxfs

            self.samples.append(sample)

        if self.sort:
            # sort the samples by their feature length
            self.samples = sorted(
                self.samples, key=lambda sample: sample['duration'])

    def __getitem__(self, index):
        sample = self.samples[index]
        utt_id = sample['utt_id']
        if not self.no_feat:
            feat_ark = sample['feat']
            feat = torch.from_numpy(kaldi_io.read_mat(feat_ark))
            feat_length = sample['length']
        else:  # raw waveform
            wav_file = sample['wav']
            if len(wav_file.split()) > 1:
                # wav_file is in command format
                source = BytesIO(run(wav_file, shell=True, stdout=PIPE).stdout)
                wav, sampling_rate = librosa.load(
                    source,
                    sr=None,  # 'None' uses the native sampling rate
                    mono=False,  # Retain multi-channel if it's there
                )
                wav = torch.from_numpy(wav).unsqueeze(0)
            else:
                wav, sampling_rate = torchaudio.load(wav_file)

            feat = wav.transpose(0, 1)
            feat_length = feat.size(0)

        if self.train:
            graphs = sample['graph']
            return feat, feat_length, utt_id, graphs
        return feat, feat_length, utt_id

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    json_path = 'data/valid_monophone.json'
    trainset = GraphDataset(json_path, no_feat=False)
    trainloader = GraphDataLoader(trainset, batch_size=2, shuffle=False)
    feat, feat_lengths, utt_ids, graphs = next(iter(trainloader))
    print(feat.size())
    print(feat_lengths)

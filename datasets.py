"""
Defines Dataset classes.
"""

import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import pickle
import matplotlib.pyplot as plt

from utils import OrderedCounter


#=============================================================================#
# Classes
#=============================================================================#


class PTB(Dataset):
    """
    Longest line: 82 words
    Vocabulary: 9877

    train:  42068 lines
    valid:  3370 lines
    test:   3761 lines
    total:  49199 lines
    """

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 82)
        self.min_occ = kwargs.get('min_occ', 3)
        # if True only sample from start of sentence
        self.enforce_sos = kwargs.get('enforce_soe', False)

        self.raw_data_path = os.path.join(data_dir, 'ptb.' + split + '.txt')
        self.data_file = 'ptb.' + split + '.json'
        self.vocab_file = 'ptb.vocab.json'

        self.meta = dict(
            samples=dict(
                train=42068,
                valid=3370,
                test=3761,
                total=49199,
            )
        )

        if create_data:
            print("Creating new %s ptb data." % split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new." %
                  (split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        seq_len = min(
            self.max_sequence_length,
            len(self.data[idx]['input']),
        )

        length = min(
            self.data[idx]['length'],
            seq_len,
        )

        if self.enforce_sos:
            i0 = 0
        else:
            I0 = self.data[idx]['length'] - length
            if I0:
                i0 = np.random.randint(I0)
            else:
                i0 = 0

        i1 = i0 + length
        pad = (self.max_sequence_length - length)
        input = self.data[idx]['input'][i0:i1] + [self.pad_idx] * pad
        target = self.data[idx]['target'][i0:i1] + [self.pad_idx] * pad

        if len(input) != self.max_sequence_length:
            raise RuntimeError("Wrong length of input")
        if len(target) != self.max_sequence_length:
            raise RuntimeError("Wrong length of target")

        return {
            'input': np.asarray(input),
            'target': np.asarray(target),
            'length': length,
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                line = line.strip()
                if not len(line):
                    continue

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length - 1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i" % (len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length - length))
                target.extend(['<pad>'] * (self.max_sequence_length - length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                line = line.strip()
                if not len(line):
                    continue

                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

    def stats(self):
        """
        return dataset stats.
        """
        all_len = sorted([d["length"] for d in self.data.values()])
        stats = {
            "k longest": all_len[-3:],
            "size": len(all_len),
            "mean": np.mean(all_len),
            "std": np.std(all_len),
        }

        return stats

    def plot(self, base_fname=None, fig_kwargs={"figsize": (3, 2)}, savefig_kwargs={"dpi": 200}):
        """
        Plot stats related information.
        """

        plots_dict = {}

        # sentences length histogram
        all_len = sorted([d["length"] for d in self.data.values()])
        plots_dict["length"] = plt.figure(**fig_kwargs)
        plt.hist(all_len, density=True, bins=50, range=(0, 260))
        plt.axvline(np.mean(all_len), lw=3, ls="--", c="k")
        plt.tight_layout()

        if base_fname is not None:
            for name, fig in plots_dict.items():
                fig.savefig(base_fname + "-" + name, **savefig_kwargs)

        return plots_dict

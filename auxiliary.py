"""
Auxiliary functions.
"""

import os
import json
import torch
import math
import numpy as np
import re

import datasets
import utils


#=============================================================================#
# Global variables
#=============================================================================#

millnames = ['', ' K', ' M', ' G', ' T']

#=============================================================================#
# Functions
#=============================================================================#


def millify(n):
    """
    Return a human readable number.
    """
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def reset_seed(seed):
    """
    Reset torch and numpy seed.
    """
    # reset seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)


def load_vocab(dataset_name, args):
    """
    Load vocabulary file.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "ptb":
        with open(os.path.join(args.data_dir, "ptb/ptb.vocab.json"), 'r') as file:
            vocab = json.load(file)
    else:
        raise ValueError("Unknown dataset = {dataset}".format(dataset=dataset_name))

    return vocab


def load_dataset(dataset_name, split, args, enforce_sos=False):
    """
    Load a split from a dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "ptb":
        dataset = datasets.PTB(
            data_dir=os.path.join(args.data_dir, "ptb/"),
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ,
            enforce_sos=enforce_sos,
        )
    else:
        raise ValueError("Unknown dataset = {dataset}".format(dataset=dataset_name))

    return dataset


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    # ind = Variable(torch.LongTensor(ind).transpose(0, 1))
    ind = utils.to_var(torch.LongTensor(ind).transpose(0, 1))
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

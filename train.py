#!/usr/bin/env python
"""
Train a latent variable language model using MIM or VAE learning.
"""

import os
import json
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from utils import to_var, idx2word
from model import SentenceMIM
import auxiliary as aux


def main(args):
    # set random seed
    aux.reset_seed(args.seed)

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    dataset_name = args.dataset.lower()

    for split in splits:
        datasets[split] = aux.load_dataset(
            dataset_name=dataset_name,
            split=split,
            args=args,
        )

    model = SentenceMIM(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        min_logv=args.min_logv,
        prior_type=args.prior_type,
        marginal=args.marginal,
        sample_mode=args.sample_mode,
        temperature=None if args.temperature <= 0.0 else args.temperature,
        reverse_input=not args.no_reverse_input,
    )

    model.args = args

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    print(model)

    ts = time.strftime('{dataset_name}_%Y-%b-%d_%H-%M-%S'.format(dataset_name=dataset_name), time.gmtime())
    if args.mim:
        ts = ts + "_mim"
    else:
        ts = ts + "_vae"

    save_model_path = os.path.join(args.save_model_path, ts)

    print("save_model_path = {save_model_path}".format(save_model_path=save_model_path))
    os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0, split):
        if split == "train":
            if anneal_function == 'logistic':
                return float(1 / (1 + np.exp(-k * (step - x0))))
            elif anneal_function == 'linear':
                return min(1, step / x0)
        else:
            # prevent annealing for val/test
            return 1.0

    nll = torch.nn.NLLLoss(reduction="sum", ignore_index=datasets['train'].pad_idx)

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0, z, split,
                pad_idx=datasets['train'].pad_idx):

        # cut-off unnecessary padding from target, and flatten
        batch_size = logv.shape[0]
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = nll(logp, target)

        if args.mim:
            # A-MIM learning
            NLL_loss = NLL_loss + np.log(batch_size)

            q_z_given_x = torch.distributions.Normal(
                loc=mean,
                scale=torch.exp(0.5 * logv),
            )

            p_z = model.get_prior()

            KL_loss = -0.5 * torch.sum(
                q_z_given_x.log_prob(z).sum(-1) +
                p_z.log_prob(z).sum(-1)
            )

            KL_weight = 1.0
        else:
            # VAE learning
            # KL Divergence
            KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
            KL_weight = kl_anneal_function(anneal_function, step, k, x0, split)

        # train as autoencoder
        if model.marginal:
            KL_weight = 0.0
            KL_loss = KL_loss.detach()

        return NLL_loss, KL_loss, KL_weight

    optim_name = args.optim.lower()
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unknown optim = {optim}".format(optim=args.optim))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    best_loss = float("inf")
    max_sequence_length = args.max_sequence_length
    scheduler_counter = args.scheduler_counter

    early_stopping = False

    for epoch in range(args.epochs):
        for split in splits:
            # update max_sequence_length
            model.max_sequence_length = max_sequence_length
            datasets[split].max_sequence_length = max_sequence_length
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == 'train',
                drop_last=True,
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout and encoder imputation
            if split == 'train':
                model.train()
            else:
                model.eval()

            # loop over dataset
            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                with torch.set_grad_enabled(split == 'train'):
                    # Forward pass
                    logp, mean, logv, z = model(batch['input'], batch['length'])

                    # loss calculation
                    NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                                                           batch['length'], mean, logv,
                                                           args.anneal_function, step, args.k, args.x0,
                                                           z, split)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer.step()
                    step += 1

                # book keeping
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.unsqueeze(0)))
                tracker['NLL'] = torch.cat((tracker['NLL'], (NLL_loss / batch_size).data.unsqueeze(0)))
                tracker['TOKENS'] = torch.cat((tracker['TOKENS'], batch['length'].sum().unsqueeze(0).type_as(loss)))

                if iteration % args.print_every == 0 or iteration + 1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, PPL %9.4f"
                          % (split.upper(), iteration, len(data_loader) - 1,
                             loss.data.item(),
                             NLL_loss.data.item() / batch_size,
                             KL_loss.data.item() / batch_size, KL_weight,
                             torch.exp(NLL_loss.data / batch['length'].sum()).item(),
                             ))

                # save checkpoint
                if (split == 'train') and (iteration % 10000 == 0):
                    checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))

            print("%s Epoch %02d/%i, Mean ELBO %9.4f, Mean NLL %9.4f, Mean PPL %9.4f, lr %e" %
                  (
                      split.upper(), epoch, args.epochs,
                      torch.mean(tracker['ELBO']),
                      torch.mean(tracker['NLL']),
                      torch.exp(torch.sum(tracker['NLL']) * batch_size / torch.sum(tracker['TOKENS'])),
                      optimizer.param_groups[0]["lr"],
                  ))

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                # save best model
                if best_loss > torch.mean(tracker['ELBO']):
                    best_loss = torch.mean(tracker['ELBO'])
                    checkpoint_path = os.path.join(save_model_path, "best.pytorch")
                    model.args.best_loss = best_loss.item()
                    model.args.best_epoch = epoch
                    torch.save(model, checkpoint_path)
                    print("Best model saved for epoch = %i at %s" % (epoch, checkpoint_path))

                else:
                    # anneal lr if no improvement
                    if scheduler_counter > 0:
                        print("Learning rate reduction counter = {scheduler_counter}".format(
                            scheduler_counter=scheduler_counter))
                        scheduler_counter -= 1
                    else:
                        scheduler.step()
                        print("\n==> Reduced learning rate lr = {lr}\n".format(lr=optimizer.param_groups[0]["lr"]))
                        scheduler_counter = args.scheduler_counter

                    if optimizer.param_groups[0]["lr"] < 1e-8:
                        print("Early stopping")
                        early_stopping = True

            if early_stopping:
                break
        if early_stopping:
            break

    print("Results path = {save_model_path}".format(save_model_path=save_model_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data/datasets')
    parser.add_argument('--dataset', type=str, default='ptb')
    # number of samples
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--scheduler_counter', type=int, default=2)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--mim', action='store_true')
    parser.add_argument('--marginal', action='store_true')
    parser.add_argument('--optim', type=str, default='adam')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-prior', '--prior_type', type=str, default='normal')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-mlv', '--min_logv', type=float, default=-10.0)
    parser.add_argument('-sm', '--sample_mode', type=str, default="sample")
    parser.add_argument('--no_reverse_input', action='store_true')

    # by default will use 1/latent_size ^ 0.5
    parser.add_argument('-temp', '--temperature', type=float, default=-1.0)
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    # kl annealing
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-bin', '--save_model_path', type=str, default='data/torch-generated/exp')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)

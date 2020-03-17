#!/usr/bin/env python
"""
Test a trained latent variable language model.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import argparse
import traceback
import numpy as np
from collections import defaultdict
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

import npeet.entropy_estimators as ee
from nltk.translate import bleu_score

from utils import to_var, idx2word, interpolate
import auxiliary as aux


def main(checkpoint_fname, args):

    #=============================================================================#
    # Load model
    #=============================================================================#

    if not os.path.exists(checkpoint_fname):
        raise FileNotFoundError(checkpoint_fname)

    model = torch.load(checkpoint_fname)

    print("Model loaded from %s" % (checkpoint_fname))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    # update max_sequence length
    if args.max_sequence_length > 0:
        model.max_sequence_length = args.max_sequence_length
    else:
        # else use model training value
        args.max_sequence_length = model.max_sequence_length

    if args.sample_mode:
        model.sample_mode = args.sample_mode

    compute_temperature = False
    if args.temperature > 0.0:
        model.temperature = args.temperature
    elif args.temperature == 0.0:
        model.temperature = 1.0 / model.latent_size * 0.5
    elif args.temperature < 0.0:
        compute_temperature = True

    model.eval()

    base_fname = "{base_fname}-{split}-{seed}-marginal{marginal}-mcmc{mcmc}".format(
        base_fname=os.path.splitext(checkpoint_fname)[0],
        split=args.split,
        seed=args.seed,
        marginal=int(model.marginal),
        mcmc=args.mcmc,
    )

    data_fname = base_fname + ".csv"
    log_fname = base_fname + ".txt"

    print("log_fname = {log_fname}".format(log_fname=log_fname))

    #=============================================================================#
    # Log results
    #=============================================================================#

    log_fh = open(log_fname, "w")

    def print_log(*args, **kwargs):
        """
        Print to screen and log file.
        """

        print(*args, **kwargs, file=log_fh)
        return print(*args, **kwargs)

    #=============================================================================#
    # Load data
    #=============================================================================#
    print_log('----------INFO----------\n')
    print_log("checkpoint_fname = {checkpoint_fname}".format(checkpoint_fname=checkpoint_fname))
    print_log("\nargs = {args}\n".format(args=args))
    print_log("\nmodel.args = {args}\n".format(args=model.args))
    params_num = len(torch.nn.utils.parameters_to_vector(model.parameters()))
    print_log("\n parameters number = {params_num_h} [{params_num}]".format(
        params_num_h=aux.millify(params_num),
        params_num=params_num,
    ))

    dataset_name = model.args.dataset.lower()
    split = args.split

    dataset = aux.load_dataset(
        dataset_name=dataset_name,
        split=split,
        args=args,
        enforce_sos=False,
    )

    vocab = aux.load_vocab(
        dataset_name=dataset_name,
        args=args,
    )
    w2i, i2w = vocab['w2i'], vocab['i2w']

    if (args.batch_size <= 0):
        args.batch_size = args.num_samples

    args.batch_size = min(len(dataset), args.batch_size)
    args.num_samples = min(len(dataset), args.num_samples)

    # collect all stats
    stats = {}

    #=============================================================================#
    # Evaluate model
    #=============================================================================#

    if args.test:
        print("Testing model")
        print("data_fname = {data_fname}".format(data_fname=data_fname))

        print_log('----------EVALUATION----------\n')

        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # entropy
        nll = torch.nn.NLLLoss(reduction="none", ignore_index=dataset.pad_idx)

        def loss_fn(logp, target, length, mean, logv, z, nll=nll, N=1,
                    eos_idx=dataset.eos_idx,
                    pad_idx=dataset.pad_idx,
                    ):
            batch_size = target.size(0)

            # do not count probability over <eos> toekn
            eos_I = (target == eos_idx)
            target[eos_I] = pad_idx

            # cut-off unnecessary padding from target, and flatten
            target = target[:, :torch.max(length).item()].contiguous().view(-1)

            # dataset size
            N = torch.tensor(N).type_as(logp)

            log_p_x_given_z = logp.view(-1, logp.size(2))

            q_z_given_x = torch.distributions.Normal(
                loc=mean,
                scale=torch.exp(0.5 * logv),
            )
            log_q_z_given_x = q_z_given_x.log_prob(z).sum(-1)

            # conditional entropy
            H_p_x_given_z = nll(log_p_x_given_z, target).view((batch_size, -1)).sum(-1)

            if model.args.mim:
                log_p_z = log_q_z_given_x - torch.log(N)
            else:
                p_z = model.get_prior()
                log_p_z = p_z.log_prob(z)

            if len(log_p_z.shape) > 1:
                log_p_z = log_p_z.sum(-1)

            # marginal entropy
            CE_q_p_z = (-log_p_z)
            H_q_z_given_x = (-log_q_z_given_x)
            # KL divergence between q(z|x) and p(z)
            KL_q_p = CE_q_p_z - H_q_z_given_x

            # NLL upper bound
            if model.args.mim:
                # MELBO
                H_p_x = H_p_x_given_z + torch.log(N)
            else:
                # ELBO
                H_p_x = H_p_x_given_z + KL_q_p

            return dict(
                H_q_z_given_x=H_q_z_given_x,
                CE_q_p_z=CE_q_p_z,
                H_p_x_given_z=H_p_x_given_z,
                H_p_x=H_p_x,
                KL_q_p=KL_q_p,
            )

        def test_model(model=model, data_loader=data_loader, desc="", plot_dist=False,
                       base_fname=base_fname, compute_temperature=False):
            """
            Compute various quantities for a model
            """
            word_count = 0
            N = len(data_loader.dataset)
            B = N // args.batch_size
            tracker = defaultdict(tensor)

            all_z = []
            for iteration, batch in tqdm(enumerate(data_loader),
                                         desc=desc,
                                         total=B,
                                         ):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])

                all_z.append(z.detach().cpu().numpy())
                # Model evaluation
                loss_dict = loss_fn(logp, batch['target'],
                                    batch['length'], mean, logv, z,
                                    N=N,
                                    )

                # aggregate values
                for k, v in loss_dict.items():
                    tracker[k] = torch.cat((tracker[k], v.detach().data))

                # subtract <eos> token from word count
                word_count = (word_count
                              + batch['length'].sum().type_as(z)
                              - (batch['target'] == dataset.eos_idx).sum().type_as(z))

                # BLEU
                if args.test_bleu:
                    recon, _, recon_l = model.decode(
                        z=model.encode(batch['input'], batch['length']),
                    )
                    batch_bleu = []
                    for d, dl, r, rl in zip(batch['target'], batch['length'],
                                            recon, recon_l):
                        cur_bleu = bleu_score.sentence_bleu(
                            references=[d[:dl].tolist()],
                            hypothesis=r[:rl].tolist(),
                            weights=(1.0,),
                        )
                        batch_bleu.append(cur_bleu)

                    tracker["BLEU"] = torch.cat((tracker["BLEU"], torch.tensor(batch_bleu)))
            if model.latent_size > 300:
                H_p_z = -1.0
            else:
                H_p_z = ee.entropy(np.concatenate(all_z[:1000], axis=0), base=np.e)
            H_normal_z = float(model.latent_size) / 2 * (1 + np.log(2 * np.pi))

            H_p_x_given_z_acc = tracker["H_p_x_given_z"].sum()
            H_p_x_acc = tracker["H_p_x"].sum()

            ppl_x_given_z = torch.exp(H_p_x_given_z_acc / word_count)
            ppl_x = torch.exp(H_p_x_acc / word_count)

            tracker["H_p_z"] = torch.tensor(H_p_z)
            tracker["H_normal_z"] = torch.tensor(H_normal_z)
            tracker["ppl_x_given_z"] = ppl_x_given_z
            tracker["ppl_x"] = ppl_x

            if plot_dist:
                print_log("Saving images and data to base_fname = {base_fname}*".format(
                    base_fname=base_fname,
                ))
                # Plot distribution of values
                for k, v in tracker.items():
                    if v.numel() > 1:
                        v = v.cpu().detach().numpy()
                        fig = plt.figure()
                        plt.hist(v, density=True, bins=50)
                        # mean
                        plt.axvline(np.mean(v), lw=3, ls="--", c="k")
                        if k == "H_p_x":
                            # sample entropy
                            plt.axvline(np.log(len(v)), lw=3, ls="-", c="k")

                        fig.savefig(base_fname + "-" + k + ".png", bbox_inches='tight')
                        plt.close(fig)
                        # save data
                        np.save(base_fname + "-" + k + ".npy", v)

            if compute_temperature:
                model.temperature = np.std(all_z)

            return {k: v.detach().mean().unsqueeze(0) for k, v in tracker.items()}

        tracker = defaultdict(tensor)
        for epoch in range(args.test_epochs):
            cur_tracker = test_model(
                model=model,
                data_loader=data_loader,
                desc="Batch [{:d} / {:d}]".format(epoch + 1, args.test_epochs),
                plot_dist=(epoch == 0),
                compute_temperature=(compute_temperature and (epoch == 0)),
            )

            for k, v in cur_tracker.items():
                tracker[k] = torch.cat([tracker[k], cur_tracker[k]])

        for k, v in tracker.items():
            v_mean = v.detach().cpu().mean().numpy()
            v_std = v.detach().cpu().std().numpy()
            print_log("{k} = {v_mean} +/- {v_std}".format(
                k=k,
                v_mean=v_mean,
                v_std=v_std,
            ))
            stats[k] = [k, v_mean, v_std]

        print_log("")

    #=============================================================================#
    # Save stats
    #=============================================================================#
    if len(stats):
        with open(data_fname, 'w') as fh:
            writer = csv.writer(fh)
            for k, row in stats.items():
                if isinstance(row, list):
                    writer.writerow(row)
                else:
                    writer.writerow([k, row])

    #=============================================================================#
    # Sample
    #=============================================================================#
    if args.test_sample:
        print_log("\n model.temperature = {temperature:e}\n".format(temperature=model.temperature))

        aux.reset_seed(args.seed)

        batches_in_samples = max(1, args.num_samples // args.batch_size)
        all_samples = []
        # all_z = []
        for b in range(batches_in_samples):
            samples, z, length = model.sample(
                n=args.batch_size,
                z=None,
                mcmc=args.mcmc,
            )

            all_samples.append(samples.detach())

        samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        print_log('----------SAMPLES----------\n')
        for s in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
            print_log("SAMP: {}\n".format(s))

    #=============================================================================#
    # Reconstruction
    #=============================================================================#
    aux.reset_seed(args.seed)

    # Reconstruct starting from <sos>
    dataset.enforce_sos = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # collect non-empty sentences
    data_iter = iter(data_loader)
    samples = {'input': [], 'target': [], 'length': []}

    for data in data_iter:
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = to_var(v)

        for i in range(args.batch_size):
            if (data["length"][i] >= args.min_sample_length) and (data["length"][i] <= args.max_sample_length):
                if args.no_unk_sample:
                    if ((data["input"][i] == dataset.unk_idx).sum() >= 1):
                        continue

                for k, v in data.items():
                    if k in samples:
                        samples[k].append(v[i])

            if len(samples["length"]) >= args.num_samples:
                break

        if len(samples["length"]) >= args.num_samples:
            break

    for k, v in samples.items():
        samples[k] = torch.stack(v)[:args.num_samples]

    z, mean, std = model.encode(samples['input'], samples['length'], return_mean=True, return_std=True)
    z = z.detach()
    mean = mean.detach()
    mean_recon, _ = model.inference(z=mean)
    mean_recon = mean_recon.detach()
    z_recon, _ = model.inference(z=z)
    z_recon = z_recon.detach()
    pert, _ = model.inference(z=z + torch.randn_like(z) * args.pert * std)
    pert = pert.detach()

    print_log('----------RECONSTRUCTION----------\n')
    for i, (d, mr, zr, p) in enumerate(zip(
        idx2word(samples["input"], i2w=i2w, pad_idx=w2i['<pad>']),
        idx2word(mean_recon, i2w=i2w, pad_idx=w2i['<pad>']),
        idx2word(z_recon, i2w=i2w, pad_idx=w2i['<pad>']),
        idx2word(pert, i2w=i2w, pad_idx=w2i['<pad>']),
    )):

        print_log("DATA: {}".format(d))
        print_log("MEAN RECON: {}".format(mr))
        print_log("Z RECON: {}".format(zr))
        print_log("Z PERT: {}".format(p))

        print_log("\n")

    #=============================================================================#
    # Interpolation
    #=============================================================================#
    if args.test_interp:
        args.num_samples = min(args.num_samples, z.shape[0])
        for i in range(args.num_samples - 1):
            z1 = z[i].cpu().numpy()
            z2 = z[i + 1].cpu().numpy()
            z_L2 = np.sqrt(np.sum((z1 - z2)**2))
            z_interp = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
            samples_interp, _ = model.inference(z=z_interp)
            sample0 = samples["input"][i:i + 1]
            sample1 = samples["input"][i + 1:i + 2]

            print_log('-------INTERPOLATION [ z L2 = {zL2:.3f} ] {src} -> {dst} -------\n\n[ {sample} ]\n'.format(
                src=i, dst=i + 1,
                sample=idx2word(sample0, i2w=i2w, pad_idx=w2i['<pad>'])[0],
                zL2=z_L2,
            ))
            print_log(*idx2word(samples_interp, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n\n')
            print_log('\n[ {sample} ]\n'.format(
                sample=idx2word(sample1, i2w=i2w, pad_idx=w2i['<pad>'])[0],
            ))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoints', metavar='N', type=str, nargs='+',
                        help='File or directory (e.g., /best.pytorch) path for checkpoint.')

    parser.add_argument('-bs', '--batch_size', type=int, default=-1)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-dd', '--data_dir', type=str, default='data/datasets')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('-ms', '--max_sequence_length', type=int, default=-1)
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-p', '--pert', type=float, default=10.0)
    parser.add_argument('-s', '--split', type=str, default="test")
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-te', '--test_epochs', type=int, default=1)
    parser.add_argument('-maxsl', '--max_sample_length', type=int, default=60)
    parser.add_argument('-minsl', '--min_sample_length', type=int, default=2)
    parser.add_argument('--no_unk_sample', action='store_true')
    parser.add_argument('--test_as_marginal', action='store_true')
    parser.add_argument('--mcmc', type=int, default=0)
    parser.add_argument('-sm', '--sample_mode', type=str, default="")
    # by default will use 1/latent_size ^ 0.5
    parser.add_argument('-temp', '--temperature', type=float, default=0.0)
    parser.add_argument('--test_bleu', action='store_true')
    parser.add_argument('--test_sample', action='store_true')
    parser.add_argument('--test_interp', action='store_true')

    args = parser.parse_args()

    for i, checkpoint_fname in enumerate(args.checkpoints):
        if os.path.isdir(checkpoint_fname):
            checkpoint_fname = os.path.join(checkpoint_fname, "best.pytorch")

        print("""
******************************************
[ {i: 3d} / {n: 3d} ] {checkpoint_fname}
******************************************
""".format(
            i=i + 1,
            n=len(args.checkpoints),
            checkpoint_fname=checkpoint_fname,
        ))

        try:
            with torch.no_grad():
                main(checkpoint_fname, args)
        except:
            traceback.print_exc()

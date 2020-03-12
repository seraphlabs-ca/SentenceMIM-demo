"""
Defines a probabilistic language auto-encoder that can be trained
with MIM or VAE learning.
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import auxiliary as aux


class SentenceMIM(nn.Module):
    """
    A probabilistic auto-encoder that can be trained with MIM learning
    and VAE learning.
    """

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout,
                 embedding_dropout, latent_size,
                 sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1,
                 min_logv=-10, prior_type="normal", marginal=False, sample_mode="greedy",
                 temperature=None, reverse_input=True):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # store parameters
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.min_logv = float(min_logv)
        self.prior_type = prior_type

        # If True, train as autoencoder
        self.marginal = marginal
        # greedy or sampling
        self.sample_mode = sample_mode
        # for sampling from prior
        if temperature is None:
            self.temperature = 1.0 / latent_size**0.5
        else:
            self.temperature = temperature

        self.reverse_input = reverse_input

        # construct model
        rnn_kwargs = {}
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError("Unknown rnn_type = {rnn_type}".format(rnn_type=rnn_type))

        self.encoder_rnn = rnn(embedding_size, hidden_size,
                               num_layers=num_layers,
                               batch_first=True, **rnn_kwargs)
        self.decoder_rnn = rnn(embedding_size + latent_size,  # + 1,
                               hidden_size,
                               num_layers=num_layers,
                               batch_first=True, **rnn_kwargs)

        self.hidden_factor = num_layers
        if rnn_type == 'lstm':
            # predict hidden and cell
            self.hidden_factor = self.hidden_factor * 2

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

        if prior_type == "normal":
            self.prior = torch.distributions.Normal(
                loc=to_var(torch.zeros(latent_size)),
                scale=to_var(torch.ones(latent_size)),
            )
        else:
            raise NotImplementedError("Unknown prior_type = {prior_type}".format(prior_type=self.prior_type))

    def get_prior(self):
        """
        Return model's prior
        """

        return self.prior

    def forward(self, input_sequence, length, z=None):
        """
        Efficient forward pass.
        """
        batch_size = input_sequence.size(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # reverse input sequence to encoder
        if self.reverse_input:
            rev_input_sequence = aux.reverse_padded_sequence(
                inputs=input_sequence,
                lengths=sorted_lengths,
                batch_first=True,
            )
        else:
            rev_input_sequence = input_sequence

        # ENCODER
        input_embedding = self.embedding(rev_input_sequence)

        torch.set_default_tensor_type('torch.FloatTensor')
        packed_input = rnn_utils.pack_padded_sequence(
            input_embedding, sorted_lengths.data.cpu(), batch_first=True)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # encoder forward pass
        if self.rnn_type == "lstm":
            _, (hidden, cell) = self.encoder_rnn(packed_input)
            hidden = hidden.transpose_(0, 1).contiguous().view((batch_size, -1))
            cell = cell.transpose_(0, 1).contiguous().view((batch_size, -1))
            hidden = torch.cat([hidden, cell], dim=-1)
        else:
            _, hidden = self.encoder_rnn(packed_input)
            hidden = hidden.transpose_(0, 1).contiguous().view((batch_size, -1))

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden).clamp(min=self.min_logv)
        std = torch.exp(0.5 * logv)

        if z is None:
            if self.marginal:
                z = mean
            else:
                z = to_var(torch.randn([batch_size, self.latent_size]))
                z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        else:
            input_embedding = self.embedding(input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        # append z and token index
        input_embedding = torch.cat([
            input_embedding,
            z.unsqueeze(1).expand(input_embedding.shape[0], input_embedding.shape[1], z.shape[1]),
        ],
            dim=-1)

        torch.set_default_tensor_type('torch.FloatTensor')
        packed_input = rnn_utils.pack_padded_sequence(
            input_embedding, sorted_lengths.data.cpu(), batch_first=True)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # decoder forward pass
        if self.rnn_type == "lstm":
            hidden, cell = hidden.chunk(2, dim=-1)
            hidden = hidden.view((batch_size, -1, self.hidden_size)).transpose_(0, 1).contiguous()
            cell = cell.view((batch_size, -1, self.hidden_size)).transpose_(0, 1).contiguous()
            outputs, _ = self.decoder_rnn(packed_input, (hidden, cell))
        else:
            hidden = hidden.view((batch_size, -1, self.hidden_size)).transpose_(0, 1).contiguous()
            outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True,
                                                       padding_value=self.pad_idx)[0]
        # total_length=total_length, padding_value=self.pad_idx)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def encode(self, input_sequence, length, return_mean=False, return_std=False):
        """
        Encodes sentences into corresponding latent codes z.
        """

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        _, reversed_idx = torch.sort(sorted_idx)

        input_sequence = input_sequence[sorted_idx]

        # reverse input sequence to encoder
        if self.reverse_input:
            rev_input_sequence = aux.reverse_padded_sequence(
                inputs=input_sequence,
                lengths=sorted_lengths,
                batch_first=True,
            )
        else:
            rev_input_sequence = input_sequence

        # ENCODER
        input_embedding = self.embedding(rev_input_sequence)

        torch.set_default_tensor_type('torch.FloatTensor')
        packed_input = rnn_utils.pack_padded_sequence(
            input_embedding, sorted_lengths.data.cpu(), batch_first=True)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # encoder forward pass
        if self.rnn_type == "lstm":
            _, (hidden, cell) = self.encoder_rnn(packed_input)
            hidden = hidden.transpose_(0, 1).contiguous().view((batch_size, -1))
            cell = cell.transpose_(0, 1).contiguous().view((batch_size, -1))
            hidden = torch.cat([hidden, cell], dim=-1)
        else:
            _, hidden = self.encoder_rnn(packed_input)
            hidden = hidden.transpose_(0, 1).contiguous().view((batch_size, -1))

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden).clamp(min=self.min_logv)
        std = torch.exp(0.5 * logv)

        if self.marginal:
            z = mean
        else:
            z = to_var(torch.randn([batch_size, self.latent_size]))
            z = z * std + mean

        z = z[reversed_idx]
        mean = mean[reversed_idx]

        ret_val = [z]
        if return_mean:
            ret_val.append(mean)
        if return_std:
            ret_val.append(std)
        if len(ret_val) == 1:
            ret_val = ret_val[0]

        return ret_val

    def decode(self, n=4, z=None):
        """
        Decodes latent codes z into corresponding sentences.
        """
        samples, z = self.inference(n=n, z=z)
        length = (samples != self.pad_idx).sum(-1)

        return samples, z, length

    def sample(self, n=4, z=None, mcmc=0):
        """
        Sampling (with optional MCMC chains)
        """
        samples, z, length = self.decode(n=n, z=z)

        # run MCMC chain
        for i in range(mcmc):
            z = self.encode(samples, length)
            samples, z, length = self.decode(n=n, z=z)

        return samples, z, length

    def inference(self, n=4, z=None):
        """
        Autoregressive sampling from the model.
        """
        temp = self.temperature
        if z is None:
            batch_size = n
            if self.prior_type == "normal":
                z = self.get_prior().sample_n(batch_size) * temp
            else:
                raise NotImplementedError("Unknown prior_type = {prior_type}".format(prior_type=self.prior_type))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        # accumulate tokens
        t = 0
        while(t < self.max_sequence_length and len(running_seqs) > 0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            input_embedding = torch.cat([
                input_embedding,
                z.unsqueeze(1).expand(input_embedding.shape[0], input_embedding.shape[1], z.shape[1]),
            ],
                dim=-1)

            bs = len(running_seqs)
            # decoder forward pass
            if self.rnn_type == "lstm":
                hidden, cell = hidden.chunk(2, dim=-1)
                hidden = hidden.view((bs, -1, self.hidden_size)).transpose_(0, 1).contiguous()
                cell = cell.view((bs, -1, self.hidden_size)).transpose_(0, 1).contiguous()

                output, (hidden, cell) = self.decoder_rnn(input_embedding, (hidden, cell))

                hidden = hidden.transpose_(0, 1).contiguous().view((bs, -1))
                cell = cell.transpose_(0, 1).contiguous().view((bs, -1))
                hidden = torch.cat([hidden, cell], dim=-1)
            else:
                hidden = hidden.view((bs, -1, self.hidden_size)).transpose_(0, 1).contiguous()

                output, hidden = self.decoder_rnn(input_embedding, hidden)

                hidden = hidden.transpose_(0, 1).contiguous().view((bs, -1))

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data.type_as(
                sequence_mask[sequence_running])
            sequence_running = sequence_idx.masked_select(sequence_mask.type(torch.bool))

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask.type(torch.bool))

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                if input_sequence.dim() < 1:
                    input_sequence = input_sequence.unsqueeze(0)
                input_sequence = input_sequence[running_seqs]

                # hidden = hidden[:, running_seqs]
                hidden = hidden[running_seqs]
                z = z[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode=None):
        """
        Auxiliary sampling method.
        """
        if mode is None:
            mode = self.sample_mode

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        elif mode == 'sample':
            sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
            sample = torch.multinomial(sample_prob, num_samples=1)
        elif mode == 'sample-no-unk':
            # reduce chances for <unk>
            dist[:, :, self.unk_idx] = dist.min()
            sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
            sample = torch.multinomial(sample_prob, num_samples=1)
        elif mode == 'greedy-no-unk':
            # prevent <unk>
            dist[:, :, self.unk_idx] = dist.min()
            _, sample = torch.topk(dist, 1, dim=-1)
        else:
            raise ValueError("Unknown sampling mode = {mode}".format(mode=mode))

        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        """
        Auxiliary sampling method.
        """
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

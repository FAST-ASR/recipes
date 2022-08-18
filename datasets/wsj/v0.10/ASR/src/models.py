# Copyright (c) Yiwen Shao

# Apache 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              kernel_sizes=None, strides=None, dilations=None, bidirectional=True, dropout=0, residual=False, **others):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU', 'TDNN-LSTM', 'TDNN-MFCC']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'TDNN':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN(in_dim, out_dim, num_layers,
                     hidden_dims, kernel_sizes, strides, dilations, dropout, residual)

    elif arch == 'TDNN-MFCC':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN_MFCC(in_dim, out_dim, num_layers,
                          hidden_dims, kernel_sizes, strides, dilations, dropout)

    elif arch == 'TDNN-LSTM':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN-LSTM')
        model = TDNNLSTM(in_dim, out_dim, num_layers, hidden_dims, kernel_sizes,
                         strides, dilations, bidirectional, dropout, residual)
    else:
        # we simply use same hidden dim for all rnn layers
        hidden_dim = hidden_dims[0]
        model = RNN(in_dim, out_dim, num_layers, hidden_dim,
                    arch, bidirectional, dropout)

    return model


class TDNNLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations,
                 bidirectional=False, dropout=0):
        super(TDNNLSTM, self).__init__()
        self.num_tdnn_layers = len(hidden_dims)
        self.num_lstm_layers = num_layers - len(hidden_dims)
        assert len(kernel_sizes) == self.num_tdnn_layers
        assert len(strides) == self.num_tdnn_layers
        assert len(dilations) == self.num_tdnn_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        # set lstm hidden_dim to the num_channels of the last cnn layer
        self.lstm_dim = hidden_dims[-1]
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(self.num_tdnn_layers)
        ])
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, self.num_lstm_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout)
        self.final_layer = nn.Linear(
            self.lstm_dim * self.num_directions, out_dim)

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        for i in range(len(self.tdnn)):
            # apply Tdnn
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        bsz = x.size(0)
        state_size = self.num_directions * \
            self.num_lstm_layers, bsz, self.lstm_dim
        h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True)  # (B, T, D)
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)  # (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


class tdnn_bn_relu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(tdnn_bn_relu, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = dilation * (kernel_size - 1) // 2
        self.dilation = dilation
        self.tdnn = nn.Conv1d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) +
            self.stride - 1
        ) // self.stride
        return out_lengths

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = self.bn(x)
        x = self.relu(x)
        x_lengths = self.output_lengths(x_lengths)
        return x, x_lengths


class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations,
                 dropout=0, residual=False):
        super(TDNN, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers
        assert len(dilations) == num_layers
        self.dropout = dropout
        self.residual = residual
        self.num_layers = num_layers
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dims[-1], out_dim, True)

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        for i in range(len(self.tdnn)):
            # apply Tdnn
            if self.residual and i > 0:  # residual starts from the 2nd layer
                prev_x = x
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = x + prev_x if (self.residual and i >
                               0 and x.size(2) == prev_x.size(2)) else x
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


class TDNN_MFCC(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout=0):
        super(TDNN_MFCC, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers
        assert len(dilations) == num_layers
        self.dropout = dropout
        self.num_layers = num_layers
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(num_layers)
        ])
        self.mfcc = torchaudio.transforms.MFCC()
        self.final_layer = nn.Linear(hidden_dims[-1], out_dim, True)

    def mfcc_output_lengths(self, in_lengths):
        hop_length = self.mfcc.MelSpectrogram.hop_length
        out_lengths = in_lengths // hop_length + 1
        return out_lengths

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        x = self.mfcc(x)
        x = x.squeeze(1)  # x of size (B, D, T)
        x_lengths = self.mfcc_output_lengths(x_lengths)
        for i in range(len(self.tdnn)):
            # apply Tdnn
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dim, rnn_type='LSTM', bidirectional=False, dropout=0):
        super(RNN, self).__init__()
        valid_rnn_types = ['LSTM', 'RNN', 'GRU']
        if rnn_type not in valid_rnn_types:
            raise ValueError("Only {0} types are supported but given {1}".format(
                valid_rnn_types, rnn_type))
        else:
            self.rnn_type = rnn_type
            if rnn_type == 'LSTM':
                self.rnn_module = nn.LSTM
            if rnn_type == 'RNN':
                self.rnn_module = nn.RNN
            if rnn_type == 'GRU':
                self.rnn_module = nn.GRU
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn_layer = self.rnn_module(self.in_dim, self.hidden_dim, self.num_layers,
                                         batch_first=True, bidirectional=bidirectional,
                                         dropout=dropout)
        self.final_layer = nn.Linear(hidden_dim * self.num_directions, out_dim)

    def forward(self, x, x_lengths):
        bsz = x.size(0)
        state_size = self.num_directions * \
            self.num_layers, bsz, self.hidden_dim
        h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)
        if self.rnn_type == 'LSTM':
            h0 = (h0, c0)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True, enforce_sorted=False)  # (B, T, D)
        x, _ = self.rnn_layer(x, h0)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)  # (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths

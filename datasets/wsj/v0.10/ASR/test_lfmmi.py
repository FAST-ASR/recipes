#!/usr/bin/env python3
# Copyright (c) Yiwen Shao

# Apache 2.0

import argparse
import os

import torch
import torch.nn.parallel

from models import get_model

from matgraph import (
    GraphDataset,
    GraphDataLoader,
    BucketSampler,
    BatchFSM,
    FSM,
    LFMMILoss,
)
from safe_gpu import safe_gpu

import kaldi_io

parser = argparse.ArgumentParser(description='PyChain test')
# Datasets
parser.add_argument('--test', type=str, required=True,
                    help='test set json file')
# Model
parser.add_argument('--exp', default='exp/tdnn',
                    type=str, metavar='PATH', required=True,
                    help='dir to load model and save output')
parser.add_argument('--model', default='model_best.pth.tar', type=str,
                    help='model checkpoint')
parser.add_argument('--results', default='posteriors.ark', type=str,
                    help='results filename')
parser.add_argument('--bsz', default=32, type=int,
                    help='test batchsize')
parser.add_argument("--no-cuda", action='store_false', help="use CPUs instead of GPU", dest="use_cuda")
parser.add_argument("--cuda", action='store_true', help="use GPU instead of CPU", dest="use_cuda")
parser.set_defaults(use_cuda=False)


args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda=args.use_cuda
if use_cuda:
    gpu_owner = safe_gpu.GPUOwner()

def main():
    # Data
    testset = GraphDataset(args.test, train=False)
    testloader = GraphDataLoader(testset, batch_size=args.bsz, num_workers=0)

    # Model
    checkpoint_path = os.path.join(args.exp, args.model)
    with open(checkpoint_path, 'rb') as f:
        if use_cuda:
            state = torch.load(f)
        else:
            state = torch.load(f, map_location=torch.device('cpu'))
        model_args = state['args']
        print("==> creating model '{}'".format(model_args.arch))
        model = get_model(model_args.feat_dim, model_args.num_targets,
                          model_args.layers, model_args.hidden_dims,
                          model_args.arch, kernel_sizes=model_args.kernel_sizes,
                          dilations=model_args.dilations,
                          strides=model_args.strides,
                          bidirectional=model_args.bidirectional,
                          dropout=model_args.dropout)
        # residual=model_args.residual if model_args.residual else False)
        print(model)

        if use_cuda:
            model = model.cuda()

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model.load_state_dict(state['state_dict'])

    output_file = os.path.join(args.exp, args.results)
    test(testloader, model, output_file, use_cuda)


def test(testloader, model, output_file, use_cuda):
    # switch to test mode
    model.eval()
    with open(output_file, 'wb') as f:
        for i, (inputs, input_lengths, utt_ids) in enumerate(testloader):
            if use_cuda:
                inputs = inputs.cuda()
            lprobs, output_lengths = model(inputs, input_lengths)
            for j in range(inputs.size(0)):
                output_length = output_lengths[j]
                utt_id = utt_ids[j]
                kaldi_io.write_mat(
                    f, (lprobs[j, :output_length, :]).cpu().detach().numpy(), key=utt_id)


if __name__ == '__main__':
    main()

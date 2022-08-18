#!/bin/env python3

import kaldi_io
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert H5 posteriors into Kaldi arks.")
parser.add_argument("input", type=str, help="H5 file with posteriors")
parser.add_argument("output", type=str, help="Kaldi ark file")

args = parser.parse_args()

with h5py.File(args.input) as f:
    data = {k: np.array(f[k]) for k in f.keys()}

with open(args.output, "wb") as f:
    for k in data.keys():
        kaldi_io.write_mat(f, data[k], key=k)

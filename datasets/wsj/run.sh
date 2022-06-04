#!/bin/bash

# Run WSJ experiment with LF-MMI implemented in MarkovModels.jl
# Author: Martin Kocour (ikocour@fit.vut.cz)

# Extract features
julia --project="./" script/feats_preparation.jl

# Prepare graphs
julia --project="./" script/graph_preparation.jl

# Train model
julia --project="./" script/model_training.jl

KALDI_ROOT=/path/to/kaldi script/kaldi_decoder.sh --stage -2 # Before, run Kaldi WSJ recipe

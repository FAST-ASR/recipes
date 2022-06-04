#!/bin/bash
# Author: Martin Kocour (ikocour@fit.vut.cz)
# Run WSJ experiment with LF-MMI implemented in MarkovModels.jl

# Extract features
julia --project="./" script/feats_preparation.jl

# Prepare graphs
julia --project="./" script/graph_preparation.jl

# Train model
julia --project="./" script/model_training.jl

KALDI_ROOT=/path/to/kaldi script/kaldi_decoder.sh --stage -2 # Before, run Kaldi WSJ recipe

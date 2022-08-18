export KALDI_ROOT=$HOME/work/tools/kaldi

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# END

if [ ! -e utils ]; then 
    ln -s $KALDI_ROOT/utils
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src":$PYTHONPATH

# export PATH=/mnt/matylda3/ikocour/tools/miniconda3/envs/pytorch1.8/bin:$PATH
export PATH=/usr/local/share/cuda/bin:/mnt/matylda3/ikocour/tools/miniconda3/envs/lfmmi/bin:$PWD:$PATH
export JULIA_CUDA_USE_BINARYBUILDER=false 

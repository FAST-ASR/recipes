#!/bin/bash

# Author: Martin Kocour (ikocour@fit.vut.cz)

set -euxo pipefail

WSJ_ROOT=$KALDI_ROOT/egs/wsj/s5
source $KALDI_ROOT/tools/config/common_path.sh
#echo $PATH && exit 0
export PATH="$WSJ_ROOT/steps:$WSJ_ROOT/utils:$PATH"

stage=0

num_targets=171 # 42 distinc phones incl. SIL NSN and SPN
beam=15
lbeam=8
acwt=0.5

source parse_options.sh

model=exp/kaldi/model
lang=exp/kaldi/lang
lang_test=exp/kaldi/lang_test_bd_tgpr
graph=exp/kaldi/graph_${lang_test#*lang_test_}
decode_dir=exp/kaldi/decode_test_${lang_test#*lang_test_}_b${beam}_lb${lbeam}_actw${acwt}

if [ $stage -le -2 ]; then
    mkdir -p "$lang/local/dict"
    sort -u lang/lexicon.txt >$lang/local/dict/lexicon.txt
    cat >$lang/local/dict/silence_phones.txt <<EOF
SIL
NSN
SPN
EOF
    grep "phone" lang/units.txt | awk '{print $1}' >$lang/local/dict/nonsilence_phones.txt
    echo "SIL" >$lang/local/dict/optional_silence.txt
    (
        lang=$(realpath $lang)
        cd $WSJ_ROOT
        utils/prepare_lang.sh $lang/local/dict "<UNK>" $lang/local/lang \
            $lang
    )

    cp -r $lang $lang_test
    cp $WSJ_ROOT/data/$(basename $lang_test)/G.fst $lang_test
fi

if [ $stage -le -1 ]; then
    mkdir -p $model
    $WSJ_ROOT/steps/chain/gen_topo.py `seq 4 $num_targets | xargs echo | tr ' ' ':'` "1:2:3" >$model/topo
    # 10 gaus per phoneme (does not matter)
    gmm-init-mono --shared-phones=$lang_test/phones/sets.int $model/topo 10 $model/0.mdl $model/tree
    copy-transition-model $model/0.mdl $model/0.trans_mdl
    ln -s 0.mdl $model/final.mdl
    mkgraph.sh --self-loop-scale 1.0 \
        $lang_test $model $graph
fi

if [ $stage -le 0 ]; then
   local/kaldi/convert_posteriors.py exp/nnet/wsj/output/test.h5 exp/nnet/wsj/output/test.ark 
fi

if [ $stage -le 1 ]; then
    mkdir -p $decode_dir
    latgen-faster-mapped --beam=$beam --lattice-beam=$lbeam --acoustic-scale=$acwt \
        --word-symbol-table=$lang_test/words.txt \
        $model/final.mdl $graph/HCLG.fst \
        ark:exp/nnet/wsj/output/test.ark \
        "ark:|lattice-scale --acoustic-scale=1.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz"
    (
        graph=$(realpath $graph)
        decode_dir=$(realpath $decode_dir)
        cd $WSJ_ROOT  
        local/score.sh data/test_eval92 $graph $decode_dir
    )
    cat $decode_dir/scoring_kaldi/best_wer
fi

if [ $stage -le 2 ]; then
    echo "Rescoring lattices."
    mkdir -p ${decode_dir}_lmrescore-fgconst
    lattice-lmrescore --lm-scale=-1.0 \
        "ark:gunzip -c $decode_dir/lat.1.gz |" \
        "fstproject --project_output=true $lang_test/G.fst |" ark:- |\
    lattice-lmrescore-const-arpa --lm-scale=1.0 ark:- \
        $WSJ_ROOT/data/lang_test_bd_fgconst/G.carpa \
        "ark,t:|gzip -c>${decode_dir}_lmrescore-fgconst/lat.1.gz"
    
    (
        graph=$(realpath $graph)
        decode_dir=$(realpath $decode_dir)
        cd $WSJ_ROOT  
        local/score.sh data/test_eval92 $graph ${decode_dir}_lmrescore-fgconst
    )
    cat ${decode_dir}_lmrescore-fgconst/scoring_kaldi/best_wer
fi

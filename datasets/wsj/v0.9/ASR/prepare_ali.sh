#!/bin/bash
train_set=train
valid_set=dev
rootdir=data
langdir=data/lang
alidir=data/ali

type=mono # mono or tri
unit=phone
model=

dumpdir=
stage=3

. path.sh
. cmd.sh
. utils/parse_options.sh

WSJ=$KALDI_ROOT/egs/wsj/s5

set -euxo pipefail

if [ -z "$model" ]; then
    case $type in
        "mono")
            model=$WSJ/exp/mono0a
            ali_script=steps/align_si.sh
            fea_script="steps/make_mfcc.sh"
            fea_type="mfcc"
            ;;
        "tri")
            model=$WSJ/exp/tri4b
            ali_script=steps/align_fmllr.sh
            fea_script="steps/make_mfcc.sh"
            fea_type="mfcc"
            ;;
#        "bi")
#            model=$WSJ/exp/
#            ali_script=
#            fea_script="steps/make_mfcc.sh"
#            fea_type="mfcc"
        *)
            echo "Type has to be one of [mono, bi, tri], got \"$type\""
            echo "Please specify correct type or model dir!"
            exit 1
            ;;
        esac
fi
echo "Using $model for alignment."

if [ $stage -le 1 ]; then
    echo "Preparing data for alignment"
    for d in $train_set $valid_set
    do
        $fea_script --nj 20 --cmd "$feats_cmd" \
            $rootdir/$d $dumpdir/$d/log $dumpdir/$d/$fea_type
        steps/compute_cmvn_stats.sh $rootdir/$d $dumpdir/$d/log $dumpdir/$d/cmvn
    done
fi

if [ $stage -le 2 ]; then
    echo "Aligning data"
    for d in $train_set $valid_set
    do
        $ali_script --nj 10 --cmd "$train_cmd" \
            $rootdir/$d $langdir $model $alidir/$d
    done

    cp $model/{cmvn_opts,final.mdl,final.occs,phones.txt,tree} $alidir
fi

if [ $stage -le 3 ]; then
    echo "Converting alignments into PDFs"
    dir="$(realpath $alidir)"
    for d in $train_set $valid_set
    do
        nj=$(cat $alidir/$d/num_jobs)
        $feats_cmd JOB=1:$nj $dir/$d/log/ali_to_pdfs.JOB.log \
            ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/$d/ali.JOB.gz |"  \
                ark,scp:$dir/$d/ali_${type}${unit}.JOB.ark,$dir/$d/ali_${type}${unit}.JOB.scp

        for n in `seq $nj`
        do
            cat $dir/$d/ali_${type}${unit}.$n.scp
        done >$dir/$d/ali_${type}${unit}.scp
    done
fi


#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

# data related
rootdir=data
dumpdir=data/dump   # directory to dump full features
featype=mfcc # mfcc or fbank
wsj0=
wsj1=

train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
train_subset_size=0
stage=0

# feature configuration
do_delta=false

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

set -euxo pipefail

feaconf="--${featype}-config conf/${featype}_hires.conf"

if [ ${stage} -le 0 ]; then
  echo "Extracting MFCC features"
  for x in $train_set $valid_set $test_set; do
    rm $rootdir/${x}/feats.scp || true
    steps/make_${featype}.sh --cmd "$train_cmd" --nj 50 \
        $feaconf $rootdir/${x}
    # compute global CMVN
    feat_dir=${dumpdir}/${x}/${featype}
    mkdir -p $feat_dir
    compute-cmvn-stats --spk2utt=ark:$rootdir/${x}/spk2utt scp:$rootdir/${x}/feats.scp ark,scp:$feat_dir/cmvn.ark,$feat_dir/cmvn.scp
  done
fi

train_feat_dir=${dumpdir}/${train_set}/${featype}; mkdir -p ${train_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}/${featype}; mkdir -p ${valid_feat_dir}
test_feat_dir=${dumpdir}/${test_set}/${featype}; mkdir -p ${test_feat_dir}
if [ ${stage} -le 1 ]; then
  echo "Dumping Features with CMVN"
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    $rootdir/${train_set}/feats.scp ark:${train_feat_dir}/cmvn.ark ${train_feat_dir}/log ${train_feat_dir}
  rm $rootdir/${train_set}/feats.scp; ln -s `realpath $train_feat_dir`/feats.scp $rootdir/${train_set}/feats.scp
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    $rootdir/${valid_set}/feats.scp ark:${valid_feat_dir}/cmvn.ark ${valid_feat_dir}/log ${valid_feat_dir}
  rm $rootdir/${valid_set}/feats.scp; ln -s `realpath $valid_feat_dir`/feats.scp $rootdir/${valid_set}/feats.scp
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    $rootdir/${test_set}/feats.scp ark:${test_feat_dir}/cmvn.ark ${test_feat_dir}/log ${test_feat_dir}
  rm $rootdir/${test_set}/feats.scp; ln -s `realpath $test_feat_dir`/feats.scp $rootdir/${test_set}/feats.scp
fi

# randomly select a subset of train set for optional diagnosis
if [ $train_subset_size -gt 0 ]; then
  train_subset_feat_dir=${dumpdir}/${train_set}_${train_subset_size}; mkdir -p ${train_subset_feat_dir}
  utils/subset_data_dir.sh $rootdir/${train_set} ${train_subset_size} $rootdir/${train_set}_${train_subset_size}
  utils/filter_scp.pl $rootdir/${train_set}_${train_subset_size}/utt2spk ${train_feat_dir}/feats.scp \
		      > ${train_subset_feat_dir}/feats.scp
fi

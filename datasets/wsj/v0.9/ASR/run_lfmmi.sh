#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

. ./cmd.sh
set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# data related
rootdir=data
dumpdir=/mnt/scratch/tmp/ikocour/PYCHAIN/WSJ   # directory to dump full features
langdir=data/lang   # directory for language models
graphdir=data/graph # directory for chain graphs (FSTs)
alidir=data/ali
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
elif [[ $(hostname -f) == *.fit.vutbr.cz ]]; then
  wsj0="/mnt/matylda2/data/WSJ/WSJ0"
  wsj1="/mnt/matylda2/data/WSJ/WSJ1"
fi

# Data splits
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92

# feature configuration
featype=mfcc
do_delta=false

# Model options
unit=phone # phone/char
type=mono # mono/bi
loss=chain # chain or cross entropy (XEnt)
checkpoint="" # path to the checkpoint

# NNet Model options
arch=TDNN
train_opts=""

affix=
nnet_affix=

# Decoding options
decode_model=model_best.pth.tar

. ./path.sh
. ./utils/parse_options.sh

[ $loss != 'chain' ] && affix="${affix:+_$affix}${loss}"

dir=exp/${arch,,}_${type}${unit}${nnet_affix:+_$nnet_affix}
lang=$langdir/lang_${type}${unit}_e2e
graph=$graphdir/${type}${unit}${affix:+_$affix}
ali=$alidir/${type}${unit}${affix:+_$affix}

echo "stage: $stage"
if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
  srcdir=data/local/data
  for x in $train_set $valid_set $test_set; do
    mkdir -p $rootdir/$x
    cp $srcdir/${x}_wav.scp $rootdir/$x/wav.scp || exit 1;
    cp $srcdir/$x.txt $rootdir/$x/text || exit 1;
    cp $srcdir/$x.spk2utt $rootdir/$x/spk2utt || exit 1;
    cp $srcdir/$x.utt2spk $rootdir/$x/utt2spk || exit 1;
    utils/filter_scp.pl $rootdir/$x/spk2utt $srcdir/spk2gender > $rootdir/$x/spk2gender || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Feature Extraction"
  ./prepare_feat.sh --wsj0 $wsj0 \
		    --wsj1 $wsj1 \
		    --train_set $train_set \
		    --valid_set $valid_set \
		    --test_set $test_set \
            --featype $featype \
		    --dumpdir $dumpdir \
		    --rootdir $rootdir
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dictionary and LM Preparation"
  ./prepare_lang.sh --langdir $langdir \
		    --unit $unit \
		    --wsj1 $wsj1
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Graph Preparation"
  ./prepare_graph.sh --train_set $train_set \
		     --valid_set $valid_set \
		     --rootdir $rootdir \
		     --graphdir $graphdir \
		     --langdir $langdir \
		     --type $type \
		     --unit $unit
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Dump Json Files"
  train_wav=$rootdir/$train_set/wav.scp
  train_dur=$rootdir/$train_set/utt2dur
  train_feat=$dumpdir/$train_set/$featype/feats.scp
  train_fst="/mnt/matylda3/ikocour/project/FAST-ASR/recipes/datasets/wsj/v0.9/data/graphs/numfsms/train/fsm.scp"
  train_text=$rootdir/$train_set/text
  train_ali=""; [ $loss != "chain" ] && train_ali=$ali/$train_set/ali_${type}${unit}.scp
  train_utt2num_frames=$rootdir/$train_set/utt2num_frames

  valid_wav=$rootdir/$valid_set/wav.scp
  valid_dur=$rootdir/$valid_set/utt2dur
  valid_feat=$dumpdir/$valid_set/$featype/feats.scp
  valid_fst="/mnt/matylda3/ikocour/project/FAST-ASR/recipes/datasets/wsj/v0.9/data/graphs/numfsms/dev/fsm.scp"
  valid_text=$rootdir/$valid_set/text
  valid_ali=""; [ $loss != "chain" ] && valid_ali=$ali/$valid_set/ali_${type}${unit}.scp
  valid_utt2num_frames=$dumpdir/$valid_set/$featype/utt2num_frames

  test_wav=$rootdir/$test_set/wav.scp
  test_dur=$rootdir/$test_set/utt2dur
  test_feat=$dumpdir/$test_set/$featype/feats.scp
  test_text=$rootdir/$test_set/text
  test_utt2num_frames=$dumpdir/$test_set/$featype/utt2num_frames
 
  set -x 
  asr_prep_json.py --wav-files $train_wav \
		   --dur-files $train_dur \
		   --feat-files $train_feat \
		   --numerator-fst-files "$train_fst" \
           --ali-files "$train_ali" \
		   --text-files $train_text \
		   --num-frames-files $train_utt2num_frames \
		   --output data/train_${type}${unit}.json
  asr_prep_json.py --wav-files $valid_wav \
		   --dur-files $valid_dur \
		   --feat-files $valid_feat \
		   --numerator-fst-files "$valid_fst" \
           --ali-files "$valid_ali" \
		   --text-files $valid_text \
		   --num-frames-files $valid_utt2num_frames \
		   --output data/valid_${type}${unit}.json
  asr_prep_json.py --wav-files $test_wav \
		   --dur-files $test_dur \
		   --feat-files $test_feat \
		   --text-files $test_text \
		   --num-frames-files $test_utt2num_frames \
		   --output data/test_${type}${unit}.json
fi


if [ ${stage} -le 5 ]; then
  echo "Stage 5: Model Training"
  opts=""
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  if [ $loss != "chain" ]; then
    num_targets=$(tree-info $ali/tree | grep num-pdfs | awk '{print $2}')
    train_script="local/nnet/train_dnn.py --strides 1 1 1 1 1" 
  else
    num_targets=$(cat ../data/graphs/numpdf)
    train_script="train_lfmmi.py --den-fst /mnt/matylda3/ikocour/project/FAST-ASR/recipes/datasets/wsj/v0.9/data/graphs/denominator.fsm"
  fi

  set -x
#$cuda_cmd log/train.log \
#    conda activate lfmmi \; \
    $train_script $train_opts \
      --train data/train_${type}${unit}.json \
      --valid data/valid_${type}${unit}.json \
      --train-bsz 32 \
      --valid-bsz 16 \
      --arch $arch \
      --epochs 80 \
      --dropout 0.2 \
      --wd 0.01 \
      --optimizer adam \
      --lr 0.0002 \
      --scheduler plateau \
      --gamma 0.5 \
      --hidden-dims 384 384 384 384 384 \
      --strides 1 1 1 1 3 \
      --curriculum 1 \
      --num-targets $num_targets \
      --seed 1 \
      --exp $dir \
      --resume "$checkpoint" 2>&1 | tee $log_file
fi

if [ ${stage} -le 6 ]; then
  echo "Stage 6: Dumping Posteriors for Test Data"
  log_file=$dir/logs/dump_${test_set}.log
  result_file=$test_set/posteriors.ark
  mkdir -p $dir/$test_set
  test_lfmmi.py \
	  --test data/test_${type}${unit}.json \
	  --model $decode_model \
	  --results $result_file \
	  --exp $dir 2>&1 | tee $log_file

  log_file=$dir/logs/dump_${valid_set}.log
  result_file=$valid_set/posteriors.ark
  mkdir -p $dir/$valid_set
  test_lfmmi.py \
	  --test data/valid_${type}${unit}.json \
	  --model $decode_model \
	  --results $result_file \
	  --exp $dir 2>&1 | tee $log_file
fi

exit 0

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Trigram LM Decoding"
  decode_dir=$dir/decode/$test_set/bd_tgpr
  mkdir -p $decode_dir
  latgen-faster-mapped --acoustic-scale=1.0 --beam=15 --lattice-beam=8 \
		       --word-symbol-table="$graph/graph_bd_tgpr/words.txt" \
		       $graph/0.trans_mdl $graph/graph_bd_tgpr/HCLG.fst \
		       ark:$dir/$test_set/posteriors.ark \
		       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz" \
		       2>&1 | tee $dir/logs/decode_${test_set}.log
fi

if [  $stage -le 8 ]; then
  echo "Stage 8: Forthgram LM rescoring"
  oldlang=$langdir/$unit/lang_${unit}_test_bd_tgpr
  newlang=$langdir/$unit/lang_${unit}_test_bd_fgconst
  oldlm=$oldlang/G.fst
  newlm=$newlang/G.carpa
  oldlmcommand="fstproject --project_output=true $oldlm |"
  olddir=$dir/decode/$test_set/bd_tgpr
  newdir=$dir/decode/$test_set/fgconst
  mkdir -p $newdir
  $train_cmd $dir/logs/rescorelm_${test_set}.log \
	     lattice-lmrescore --lm-scale=-1.0 \
	     "ark:gunzip -c ${olddir}/lat.1.gz|" "$oldlmcommand" ark:- \| \
	     lattice-lmrescore-const-arpa --lm-scale=1.0 \
	     ark:- "$newlm" "ark,t:|gzip -c>$newdir/lat.1.gz"
fi

if [ ${stage} -le 9 ]; then
  echo "Stage 9: Computing WER"
  for lmtype in bd_tgpr fgconst; do
    local/score_kaldi_wer.sh $rootdir/$test_set $graph/graph_bd_tgpr $dir/decode/$test_set/$lmtype
    echo "Best WER for $dataset with $lmtype:"
    cat $dir/decode/$test_set/$lmtype/scoring_kaldi/best_wer
  done
fi

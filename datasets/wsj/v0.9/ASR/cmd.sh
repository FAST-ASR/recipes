# Default config for BUT cluster,

export feats_cmd="queue.pl --mem 2G"
export train_cmd="queue.pl --mem 1G"
export decode_cmd="queue.pl --mem 2G"
export cuda_cmd="queue.pl --gpu 1 --mem 20G"

if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  queue_conf=$PWD/cmd.conf # see example /homes/kazi/iveselyk/queue_conf/default.conf,
  export feats_cmd="queue.pl --config $queue_conf --mem 2G --matylda 1"
  export train_cmd="queue.pl --config $queue_conf --mem 1.9G --matylda 0.5"
  export decode_cmd="queue.pl --config $queue_conf --mem 6G --matylda 0.25"
  export graph_cmd="queue.pl --config $queue_conf --mem 32G"
  #export cuda_cmd="queue.pl --config $queue_conf --gpu 1 --mem 40G --tmp 40G"
  export cuda_cmd="queue.pl --config $queue_conf --gpu 1 --mem 6G --tmp 12G " # PC0204
fi


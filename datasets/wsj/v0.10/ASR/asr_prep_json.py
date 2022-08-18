#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import logging
import sys


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


def read_file(ordered_dict, key, dtype, *paths):
    for path in paths:
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, val = line.strip().split(None, 1)
                if val[-1] == '|':
                    val = val[:-2]
                if utt_id in ordered_dict:
                    assert key not in ordered_dict[utt_id], \
                        "Duplicate utterance id " + utt_id + " in " + key
                    ordered_dict[utt_id].update({key: dtype(val)})
                else:
                    ordered_dict[utt_id] = {key: dtype(val)}
    return ordered_dict


def read_file_orig(ordered_dict, key, dtype, *paths):
    tmpd = OrderedDict()
    tmpd = read_file(tmpd, key, dtype, *paths)
    err = 0
    for obj in ordered_dict.values():
        assert "orig_utts" in obj
        try:
            obj[key] = [tmpd[utt][key] for utt in obj["orig_utts"]]
        except KeyError as e:
            print(f"WARNING: Could not find key: {e} in {paths}", file=sys.stderr)
            err +=1

    if err/len(ordered_dict) >= 0.01:
        raise Exception(f"Found more than 1% errors (err: {err}) in {paths}")

    return ordered_dict


def main():
    parser = argparse.ArgumentParser(
        description="Wrap all related files of a dataset into a single json file"
    )
    # fmt: off
    parser.add_argument("--wav-files", nargs="+", required=True,
                        help="path(s) to scp raw waveform file(s)")
    parser.add_argument("--dur-files", nargs="+", required=True,
                        help="path(s) to utt2dur file(s)")
    parser.add_argument("--feat-files", nargs="+", default=None,
                        help="path(s) to scp feature file(s)")
    parser.add_argument("--num-frames-files", nargs="+", default=None,
                        help="path(s) to utt2num_frames file(s)")
    parser.add_argument("--text-files", nargs="+", default=None,
                        help="path(s) to text file(s)")
    parser.add_argument("--numerator-fst-files", nargs="+", default=None,
                        help="path(s) to numerator fst file(s)")
    parser.add_argument("--numerator-fst-orig", nargs="+", default=None,
                        help="path(s) to numerator fst file(s) of mixed speech")
    parser.add_argument("--ali-files", nargs="+", default=None,
                        help="path(s) to alignments")
    parser.add_argument("--text-files-orig", nargs="+", default=None,
                        help="path(s) to text file(s) of mixed speech")
    parser.add_argument("--ali-files-orig", nargs="+", default=None,
                        help="path(s) to alignments for mixed speech")
    parser.add_argument("--output", required=True, type=argparse.FileType("w"),
                        help="path to save json output")
    parser.add_argument("--mix2orig-utts", default=None,
                        help="map from speech mixture to original utt_ids")
    args = parser.parse_args()
    print(args)
    # fmt: on

    obj = OrderedDict()
    obj = read_file(obj, "wav", str, *(args.wav_files))
    obj = read_file(obj, "duration", float, *(args.dur_files))
    if args.mix2orig_utts:
        obj = read_file(obj, "orig_utts", lambda x: x.split(), args.mix2orig_utts)
    if args.feat_files:
        obj = read_file(obj, "feat", str, *(args.feat_files))
    if args.text_files:
        obj = read_file(obj, "text", str, *(args.text_files))
    if args.text_files_orig:
        obj = read_file_orig(obj, "text", str, *(args.text_files_orig))
    if args.numerator_fst_files:
        obj = read_file(obj, "numerator_fst", str, *(args.numerator_fst_files))
    if args.numerator_fst_orig:
        obj = read_file_orig(obj, "numerator_fst", str, *(args.numerator_fst_orig))
    if args.ali_files:
        obj = read_file(obj, "ali", str, *(args.ali_files))
    if args.ali_files_orig:
        obj = read_file_orig(obj, "ali", str, *(args.ali_files_orig))
    if args.num_frames_files:
        obj = read_file(obj, "length", int,
                        *(args.num_frames_files))

    json.dump(obj, args.output, indent=4)


if __name__ == "__main__":
    main()

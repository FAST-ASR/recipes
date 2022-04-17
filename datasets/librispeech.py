# SPDX-License-Identifier: MIT

import argparse
from lhotse.recipes import download_librispeech, prepare_librispeech
from lhotse.recipes import download_musan, prepare_musan

def main(args):
    libri_version = 'librispeech' if args.full else 'mini_librispeech'
    corpus_dir = download_librispeech(args.outdir, dataset_parts=libri_version)
    manifest = prepare_librispeech(corpus_dir, dataset_parts=libri_version,
            output_dir=args.outdir, num_jobs=args.num_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-jobs', type=int, default=1,
            help='number of parallel jobs')
    parser.add_argument('-f', '--full', action='store_true',
            help='prepare the full Librispeech corpus')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    main(args)



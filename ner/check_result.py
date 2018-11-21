# encoding=utf-8

import argparse as _ap

import data_utils as _du

def add_arguments(parser):
    parser.add_argument('--label_file', type=str, default='', help='standard anwsers.')
    parser.add_argument('--infer_file', type=str, default='', help='inferences from your own model.')


def check_r(label_file='', infer_file=''):
    return _du.compare_targets(label_file, infer_file)


if __name__ == '__main__':
    arg_parser = _ap.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS, _ = arg_parser.parse_known_args()
    _du.compare_targets(FLAGS.label_file, FLAGS.infer_file)
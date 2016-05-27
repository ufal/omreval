#!/usr/bin/env python
"""This is a script that measures the edit distance between MusicXML
trees. Currently supports the following metrics (-m):

* zss_unfiltered
* zss

Descriptions of individual metrics:


zss_unfiltered
--------------

Replacement cost: 0 if the nodes are identical, 1 if they are not.
No MusicXML tree preprocessing.


zss
---

Certain nodes are skipped when generating child lists:

        'work',
        'identification',
        'defaults',
        'credit',
        'print',
        'midi-instrument',
        'midi-device',
        'duration'


zss_Levenshtein
---------------



"""
from __future__ import print_function

import StringIO
import argparse
import logging
import time
from lxml import etree
from Levenshtein import distance

__version__ = "0.0.2"
__author__ = "Jan Hajic jr."

####################################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store',
                        help='The true MusicXML file.')
    parser.add_argument('-p', '--prediction', action='store',
                        help='The predicted (recognized) MusicXML file.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    if args.true == args.prediction:
        print('0')
        return

    true = etree.parse(args.true)
    prediction = etree.parse(args.prediction)

    _parse_time = time.clock()
    logging.info('Parsing done in: {0:.3f} s'.format(_parse_time - _start_time))

    # Argument order: "How much does it cost to turn prediction into the true tree?"
    true_output = StringIO.StringIO()
    true.write_c14n(true_output)
    true_str = true_output.getvalue()

    pred_output = StringIO.StringIO()
    prediction.write_c14n(pred_output)
    pred_str = pred_output.getvalue()

    dist = distance(true_str, pred_str)
    print('{0}'.format(dist))

    _end_time = time.clock()
    _eval_time = _end_time - _parse_time

    # Logging timing:
    n_true_notes = len(list(true.iter('note')))
    n_pred_notes = len(list(prediction.iter('note')))

    logging.info('Timing:')
    logging.info('True notes: {0}, eval. took {1:.4f} s per true note.'
                 ''.format(n_true_notes, _eval_time / n_true_notes))
    logging.info('Pred notes: {0}, eval. took {1:.4f} s per pred note.'
                 ''.format(n_pred_notes, _eval_time / n_pred_notes))

    logging.info('musicxml_eval done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

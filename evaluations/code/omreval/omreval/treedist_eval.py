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
import argparse
import logging
import time
from lxml import etree

from zss import distance

from zss_metrics import ZSSMetricClass, Lxml4ZSS, Lxml4ZSS_Filtered, Lxml4ZSS_Levenshtein
from pitch_counter import NoteContentCoder, encode_notes

__version__ = "0.0.2"
__author__ = "Jan Hajic jr."

####################################################################################

supported_metrics = {
    'zss_unfiltered': Lxml4ZSS,
    'zss': Lxml4ZSS_Filtered,
    'zss_Levenshtein': Lxml4ZSS_Levenshtein,
}


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store',
                        help='The true MusicXML file.')
    parser.add_argument('-p', '--prediction', action='store',
                        help='The predicted (recognized) MusicXML file.')
    parser.add_argument('-m', '--metric', action='store', default='zss',
                        help='The metric to use for evaluation. Currently'
                             ' supported: \'zss\', \'zss_unfiltered\','
                             '\'zss_Levenshtein\'. '
                             'Default: \'zss\'')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    if args.metric not in supported_metrics:
        raise ValueError('Metric {0} is not supported.\nSupported metrics:\n\t{1}'
                         ''.format(args.metric, '\n\t'.join(supported_metrics.keys())))
    metric_class = supported_metrics[args.metric]

    if args.true == args.prediction:
        print('0')
        return

    true = etree.parse(args.true).getroot()
    prediction = etree.parse(args.prediction).getroot()

    if args.metric == 'zss_Levenshtein':
        coder = NoteContentCoder()
        true = encode_notes(true, coder)
        prediction = encode_notes(prediction, coder)

    # Preprocess trees: only retain relevant.
    #  - part-list
    #  - part
    # From those, filter:
    #  - midi-instrument
    #  - midi-device
    #  - print

    _parse_time = time.clock()
    logging.info('Parsing done in: {0:.3f} s'.format(_parse_time - _start_time))

    # Argument order: "How much does it cost to turn prediction into the true tree?"
    dist = distance(prediction, true,
                    get_children=metric_class.get_children,
                    update_cost=metric_class.update,
                    insert_cost=metric_class.insert,
                    remove_cost=metric_class.remove
                    )

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

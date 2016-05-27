#!/usr/bin/env python
"""This is a script that evaluates MusicXML data by converting
them to LilyPond and simply computing the Levenshtein distance.

Note that this may be inaccurate as it doesn't account for copy/paste
operations."""
from __future__ import print_function
import argparse
import codecs
import logging
import os
import shlex
import subprocess
import time

import Levenshtein
import re

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def execute(command):
    """Runs this command through subprocess.popen() with pipes.
    Captures stdout."""
    logging.debug('execute("{0}")'.format(command))
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = process.communicate()[0]
    return result


def normalize_accidentals(line):
    """Substitute all sharps (-is) for an extra 'I', flats (-es, es, as)
    for an extra 'E'."""
    line = line.replace('cis', 'cI')
    line = line.replace('dis', 'dI')
    line = line.replace('eis', 'eI')
    line = line.replace('fis', 'fI')
    line = line.replace('gis', 'gI')
    line = line.replace('ais', 'aI')
    line = line.replace('his', 'hI')
    line = line.replace('bis', 'bI')

    line = line.replace('ces', 'cE')
    line = line.replace('des', 'dE')
    line = line.replace('ees', 'eE')
    line = line.replace('fes', 'fE')
    line = line.replace('ges', 'gE')
    line = line.replace('aes', 'aE')
    line = line.replace('hes', 'hE')
    line = line.replace('bes', 'bE')

    return line


def postprocess_ly(ly):
    ll = [l for l in ly.split('\n') if not l.strip().startswith('%')]
    ll = [normalize_accidentals(l) for l in ll]
    all = ' '.join(ll)

    all = all.replace('\t', ' ')
    all = all.replace('\n', ' ')

    spaces = re.compile('  *')
    all = spaces.sub(' ', all)

    return all


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store',
                        help='The true MusicXML file.')
    parser.add_argument('-p', '--prediction', action='store',
                        help='The predicted (recognized) MusicXML file.')
    parser.add_argument('--m2ly_path', action='store',
                        default='/Applications/LilyPond.app/Contents/Resources/bin/',
                        help='Directory containing musicxml2ly executable.')
    parser.add_argument('--export_pred', action='store_true',
                        help='If set, exports the lilypond conversion of the prediction '
                             '(with a *.ly suffix).')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    if not os.path.isfile(args.true):
        raise OSError('True file not found: {0}'.format(args.true))
    if not os.path.isfile(args.prediction):
        raise OSError('Prediction file not found: {0}'.format(args.prediction))

    conversion_script = os.path.join(args.m2ly_path, 'musicxml2ly')
    if not os.path.isfile(conversion_script):
        raise OSError('Cannot find musicxml2ly conversion script: {0}'.format(conversion_script))

    conversion_cmd = '{0} --lxml -a -o - '.format(conversion_script)

    true_ly = postprocess_ly(execute(conversion_cmd + args.true))
    pred_ly = postprocess_ly(execute(conversion_cmd + args.prediction))

    edits = Levenshtein.editops(true_ly, pred_ly)
    print('{0}'.format(len(edits)))

    if args.export_pred:
        with codecs.open(args.prediction + '.ly', 'w', 'utf-8') as export_h:
            export_h.write(pred_ly + u'\n')

    _end_time = time.clock()
    logging.info('lilypond_eval.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

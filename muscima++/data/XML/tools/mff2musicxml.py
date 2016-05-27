#!/usr/bin/env python
"""This script converts MFF-MUSCIMA XML data back into MusicXML,
simply by removing all eid="value" attributes.

Usage:

    mff2musicxml.py <file-mff.xml >file-musicxml.xml

"""
from __future__ import print_function
import argparse
import logging
import re
import sys
import time

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    eid_pattern = ' eid="[^"]*"'
    eid_regex = re.compile(eid_pattern)

    for line in sys.stdin:
        line_out = re.sub(eid_regex, '', line)
        sys.stdout.write(line_out)

    _end_time = time.clock()
    logging.info('main() done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

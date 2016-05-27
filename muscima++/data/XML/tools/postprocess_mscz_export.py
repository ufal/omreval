#!/usr/bin/env python
"""This script performs postprocessing modifications for
MuseScore MusicXML export. The changes that are automated:

* Default barlines made explicit
* Add <staves>1</staves> in measure.attributes
* Add explicit <staff-details> to <attributes> for 5-line staves
  for binding staff lines

Some changes must be done manually, because they do not get
exported at all:

* MuseScore doesn't export <measure-repeat> style, just <forward>
* Repeat bar sign (most pertinent in orchestral parts): gets
  exported as a whole-bar <forward>, but should get its own
  measure-style="repeat"

"""
from __future__ import print_function
import argparse
import logging
import time
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from lxml.etree import Element
#import lxml.etree as ElementTree

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."

DEFAULT_BARLINE_TEXT = '''
<barline location="right">
    <bar-style>regular</bar-style>
    </barline>

'''
default_barline_element = ElementTree.ElementTree(ElementTree.fromstring(DEFAULT_BARLINE_TEXT)).getroot()


DEFAULT_N_STAVES_TEXT = '''
<staves>1</staves>

'''
default_staves_element = ElementTree.ElementTree(ElementTree.fromstring(DEFAULT_N_STAVES_TEXT)).getroot()


DEFAULT_STAFF_DETAILS_TEXT = '''
<staff-details number="1">
    <staff-type>regular</staff-type>
    <staff-lines>5</staff-lines>
    </staff-details>

'''
default_staff_details_element = ElementTree.ElementTree(ElementTree.fromstring(DEFAULT_STAFF_DETAILS_TEXT)).getroot()


def process_barlines(root):
    """Add a default barline at the end of each measure that doesn't
    have a barline specified.

    :type root: Element
    :param root: Root of the MusicXML tree.
    """
    for e in root.iter('measure'):
        if not e.findall('barline'):
            e.append(default_barline_element)
        else:
            logging.info('Found barline.')

    return root


def process_n_staves(root):
    """Add a <staves> element to each part that doesn't define it
    because it's a default 1-staff part.

    :type root: Element
    :param root: Root of the MusicXML tree.
    """
    for e in root.iter('attributes'):
        if not e.findall('staves'):
            insert_staves_at = 0
            if e.findall('divisions'):
                insert_staves_at += len(e.findall('divisions'))
            if e.findall('key'):
                insert_staves_at += len(e.findall('key'))
            if e.findall('time'):
                insert_staves_at += len(e.findall('time'))
            e.insert(insert_staves_at, default_staves_element)
        else:
            logging.info('Found staves.')

    return root


def process_staff_details(root):
    """Add the default <staff-details>: staves have 5 lines unless
    said otherwise."""
    for e in root.iter('attributes'):
        if not e.findall('staff-details'):
            insert_staves_at = 0
            if e.findall('divisions'):
                insert_staves_at += len(e.findall('divisions'))
            if e.findall('key'):
                insert_staves_at += len(e.findall('key'))
            if e.findall('time'):
                insert_staves_at += len(e.findall('time'))
            if e.findall('staves'):
                insert_staves_at += len(e.findall('staves'))
            if e.findall('part-symbol'):
                insert_staves_at += len(e.findall('part-symbol'))
            if e.findall('instruments'):
                insert_staves_at += len(e.findall('instruments'))
            if e.findall('clef'):
                insert_staves_at += len(e.findall('clef'))
            e.insert(insert_staves_at, default_staff_details_element)
        else:
            logging.info('Found staff details.')

    return root


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store',
                        help='The input MusicXML file.')
    parser.add_argument('-o', '--output', action='store',
                        help='The output MFF-MUSCIMA XML file.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    etree = ElementTree.parse(args.input)
    root = etree.getroot()

    # Process barlines.
    root = process_barlines(root)

    # Process number of staves.
    root = process_n_staves(root)

    # Add explicit staff details to attributes for 5-line staves.
    root = process_staff_details(root)

    etree.write(args.output)

    # Your code goes here
    #print('No automated postprocessing as of yet.')

    _end_time = time.clock()
    logging.info('postprocess_mscz_export.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

#!/usr/bin/env python
"""This script converts MusicXML data into MFF-MUSCIMA XML,
adding an eid="..." attribute to each element. The ID values are
detailed below.

Usage:

    musicxml2mff.py -i file-musicxml.xml >file-mff.xml

Element IDs
-----------

An Element ID (eid) attribute is assigned to every element in the XML.
The IDs have the following format:

  [prefix_]etype-enumber-gnumber

The prefix can be specified through the ``--prefix`` (``-p``) argument.
``etype`` is just the element type (this makes it easier to manually align
to the right element from other MFF-MUSCIMA entity levels: notes get aligned
to eIDs that contain "note", etc.).

``enumber`` is the element number: this is the ``enumber``-th element of
type ``etype``. ``gnumber`` is the global element number: this is the
``gnumber``-th element overall. The numbers have no fixed width and start
at 0.
"""
from __future__ import print_function
import argparse
import logging
import time
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class EIDGenerator(object):
    """This class is responsible for generating the element IDs.
    """
    def __init__(self, prefix=None):
        self.gnumber = 0
        self.enumbers = {}

        self.prefix = prefix

    def get_eid(self, node):
        """Gets an XML element node and returns the current ID
        for it.

        :type node: Element
        :param node: The element we're generating an eID for.
        """
        tag = node.tag
        if tag not in self.enumbers:
            self.enumbers[tag] = 0

        eid = tag + '-' + str(self.enumbers[tag]) + '-' + str(self.gnumber)
        if self.prefix is not None:
            eid = self.prefix + '_' + eid

        self.gnumber += 1
        self.enumbers[tag] += 1

        return eid


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store',
                        help='The input MusicXML file.')
    parser.add_argument('-o', '--output', action='store',
                        help='The output MFF-MUSCIMA XML file.')
    parser.add_argument('-p', '--prefix', action='store',
                        help='An optional prefix for all element IDs in the document.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    eid_generator = EIDGenerator(prefix=args.prefix)

    etree = ElementTree.parse(args.input)
    root = etree.getroot()

    for e in root.iter('*'):
        eid = eid_generator.get_eid(e)
        e.attrib['eid'] = eid

    etree.write(args.output)

    _end_time = time.clock()
    logging.info('musicxml2mff.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

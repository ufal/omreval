#!/usr/bin/env python
"""This is an elementary pitch correctness evaluation
script. It has the following capabilities::

    ... | part   | voice  | staff
    --------------------------------
    YES | single | single | single
    NO  | single | multi  | single
    NO  | single | multi  | multi
    NO  | multi  | single | single
    NO  | multi  | multi  | single
    NO  | multi  | multi  | multi

We don't have a separate single-voice, multi-staff case:
once we start supporting multiple staves, we already do support
multiple voices.

Computing pitch sequence similarity
-----------------------------------

Without a move/swap operation, the distance is straightforward
Levenshtein on the pitch sequence.

With a move/swap operation, the computation becomes much more
complicated.
"""
from __future__ import print_function
import argparse
import logging
import time

import numpy
from lxml import etree

import Levenshtein

#from xml.etree import ElementTree
#from xml.etree.ElementTree import Element

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class PitchCoder(object):
    """Encode a sequence of pitches as characters for Levenshtein distance
    computation. Assumes there are no more than 92 distinct pitches
    in a composition, which should hold for all music that doesn't have
    a ridiculous amount of enharmonic changes and/or double accidentals.

    Note that the PitchCoder should be consistent throughout a composition.

    Note: capital R is reserved for rests in note coding, so it's not a part
    of pitch encoding letters.
    """
    code_letters = '0123456789' + \
                   'abcdefghijklmnopqrstuvwxyz' + \
                   'ABCDEFGHIJKLMNOPQ' + 'STUVWXYZ' + \
                   '.,!?:;/\\|-_=+><[]{}()*&^%$#@~`'

    def __init__(self):
        self.n_pitches = 0
        self.codes = dict()
        self.inverse_codes = dict()

    def pitch2pitch_index(self, pitch):
        """
        Converts a pitch element into a (step, alteration, octave) triplet
        that can be used to index the codes dict.

        :param pitch: A ``<pitch>`` element.

        :return: A hashable representation.
        """
        pitch_values = {'step': 'C',
                         'alteration': 0,
                         'octave': 0}
        for e in pitch.iter('step', 'alteration', 'octave'):
            pitch_values[e.tag] = e.text
        pitch_index = tuple(pitch_values.values())
        return pitch_index

    def encode(self, pitch):
        """Converts the pitch element into a number.

        :type pitch: etree.Element
        :param pitch: A <pitch> MusicXML lxml.etree.Element

        :return: A letter.
        """
        pitch_index = self.pitch2pitch_index(pitch)

        if pitch_index not in self.codes:
            if self.n_pitches == len(self.code_letters):
                raise ValueError('Too many distinct entities to encode, only {0}'
                                 ' codes available.'.format(len(self.code_letters)))
            self.codes[pitch_index] = self.code_letters[self.n_pitches]
            self.n_pitches += 1

        return self.codes[pitch_index]

    def decode(self, code):
        """Converts the code back into a pitch element."""
        if code not in self.inverse_codes:
            raise ValueError('Code {0} not known.'.format(code))

        pitch_index = self.inverse_codes[code]
        pitch = self.pitch_index2pitch(pitch_index)

        return pitch

    def pitches2string(self, pitches):
        """Converts the sequence of pitches into a string.

        :param pitches: A list of <pitch> elements.

        :return: The encoded string.
        """
        code = u''.join([self.encode(p) for p in pitches])
        return code

    def pitch_index2pitch(self, pitch_index):
        """Converts a pitch values triplet used as key for pitch
        encoding into the XML element."""
        if pitch_index[1] == 0:
            pitch_string = '''<pitch>
                <step>{0}</step>
                <octave>{1}</octave>
            </pitch>'''.format(pitch_index[0], pitch_index[2])
        else:
            pitch_string = '''<pitch>
                <step>{0}</step>
                <alter>{1}</alter>
                <octave>{2}</octave>
            </pitch>'''.format(pitch_index[0], pitch_index[1], pitch_index[2])
        pitch = etree.fromstring(pitch_string)
        return pitch


class NoteContentCoder(object):
    """Encodes some content of a <note> MusicXML element as a string
    to be compared using Levenshtein distance.

    The following entities are encoded:

    * <pitch> (using the PitchCoder class, or 'R' for rests)
    * <voice> (assumes voices are single-digit)
    * <type> as a single digit: 0 for 128th or shorter, 1 for 64th, 2 for 32,
      3 for 16, 4 for 8th, 5 for 4th, 6 for half, 7 for whole
    * <stem> as U/D

    The <notation> and <direction> elements are left as elements.

    Should also implement:

    * tuplets

    One position in the encoding should correspond roughly to one choice
    that needs to be made upon inserting the note.
    """
    note_type_table = {
        '32nd': 2,
        '16th': 3,
        'eighth': 4,
        'quarter': 5,
        'half': 6,
        'whole': 7,
    }

    inverse_note_type_table = {
        2: '32nd',
        3: '16th',
        4: 'eighth',
        5: 'quarter',
        6: 'half',
        7: 'whole',
    }

    stem_type_table = {
        'up': 'U',
        'down': 'D',
        None: '-',
    }

    inverse_stem_type_table = {
        'U': 'up',
        'D': 'down',
        '-': None,
    }

    REST_CODE = 'R'

    ENCODES_TAGS = ['pitch', 'voice', 'type', 'stem']

    def __init__(self):
        self.pitch_coder = PitchCoder()

    def encode(self, note):
        if len(list(note.iter('rest'))) != 0:
            p = self.REST_CODE
        else:
            p = self.pitch_coder.encode(note.iter('pitch').next())
        v = note.iter('voice').next().text
        t = self.note_type_table[note.iter('type').next().text]
        try:
            s = self.stem_type_table[note.iter('stem').next().text]
        except StopIteration:
            s = self.stem_type_table[None]

        code = '{0}{1}{2}{3}'.format(p, v, t, s)
        return code

    def decode(self, note_code):
        p = note_code[0]
        v = note_code[1]
        t = note_code[2]
        s = note_code[3]

        pitch = self.pitch_coder.decode(p)
        voice = '<voice>{0}</voice>'.format(v)
        note_type = '<type>{0}</type>'.format(self.inverse_note_type_table[t])
        stem = '<stem>{0}</stem>'.format(self.inverse_stem_type_table[s])

        return pitch, voice, note_type, stem


def encode_notes(root, coder):
    """Uses the NoteContentCoder scheme to change <note> nodes.

    :type root: etree.Element
    :param root: The root of the MusicXML tree.

    :type coder: NoteContentCoder
    :param coder: The encoder used to convert <note> nodes to
        the simplified version with pitch, voice, type and stem
        converted to text.
    """
    for note in root.iter('note'):
        code = coder.encode(note)
        note.text = code
        to_remove = [e for e in note.getchildren() if e.tag in coder.ENCODES_TAGS]
        for e in to_remove:
            note.remove(e)

    return root


def pitch_sequence_edits(true_pitches, pred_pitches):
    """Given two lists of <pitch> elements, computes their edit
    distance.

    :param pitches1: First sequence of pitches.
    :param pitches2: Second sequence of pitches.

    :return: The Levenshtein edits.
    """
    coder = PitchCoder()

    true_code = coder.pitches2string(true_pitches)
    pred_code = coder.pitches2string(pred_pitches)

    edits = Levenshtein.editops(true_code, pred_code)
    return edits


def part_pitch_distances(parts1, parts2, correction=None):
    """Computes the edit distances between each pair of parts'
    pitch sequence. Multi-voice parts are not supported (may
    produce inaccurate results).
    """
    distances = numpy.zeros((len(parts1), len(parts2)))
    p_pitches = [list(p.iter('pitch')) for p in parts1]
    q_pitches = [list(q.iter('pitch')) for q in parts2]
    for i, p in enumerate(p_pitches):
        for j, q in enumerate(q_pitches):
            distances[i, j] = len(pitch_sequence_edits(p, q))
            print('Pitches {0} (len: {1}) vs {2} (len: {3}): {4} edits'
                  ''.format(parts1[i].attrib['id'], len(p),
                            parts2[j].attrib['id'], len(q),
                            distances[i,j]))
    return distances


def match_parts_by_pitch(parts1, parts2):
    """Given two lists of parts, matches which part
    corresponds to which based on pitch sequence edit distance.

    Currently assumes there are as many parts in parts1 as in parts2.

    :param parts1: The first list of part elements.

    :param parts2: The second list of part elements.

    :return: A list of tuples with part indexes.
    """
    distances = part_pitch_distances(parts1, parts2)

    raise NotImplementedError()


class MusicXMLBrowser(object):
    """Implements useful MusicXML handling shortcuts,
    such as collecting parts.
        """

    def __init__(self, tree):
        """
        :type tree: etree.ElementTree
        :param tree: The MusicXML tree.
        """
        self.tree = tree
        self.root = tree.getroot()

        self._parts = None
        self._parts_dict = None

    @property
    def parts(self):
        """
        :return: A list of ``<part>`` elements.
        """
        if self._parts is None:
            parts = list(self.root.iter('part'))
            self._parts = parts

        return self._parts

    @property
    def parts_dict(self):
        """
        :return: A dict of parts, indexed by their ids.
        """
        if self._parts_dict is None:
            self._parts_dict = {p.attrib['id']: p for p in self.parts}

        return self._parts_dict

    @property
    def parts_ids(self):
        """
        :return: The IDs of the parts.
        """
        return self.parts_dict.keys()

    def collect_pitch_sequence(self, part_id):
        """Collects the sequence of pitch elements from the given part.

        :param part_id: The ID of the part you want to get pitches from.
        """
        if part_id not in self.parts_dict:
            raise ValueError('Part ID {0} not found. Available IDs: {1}'
                             ''.format(part_id, self.parts_dict.keys()))

        part = self.parts_dict[part_id]
        pitches = list(part.iter('pitch'))
        return pitches


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

    # Load xmls & collect pitches
    true = etree.parse(args.true)
    t_doc = MusicXMLBrowser(true)
    if len(t_doc.parts) != 1:
        raise ValueError('Can only deal with single-part scores,'
                         ' true has {0}.'.format(len(t_doc.parts)))

    prediction = etree.parse(args.prediction)
    p_doc = MusicXMLBrowser(prediction)
    if len(p_doc.parts) != 1:
        raise ValueError('Can only deal with single-part scores,'
                         ' true has {0}.'.format(len(p_doc.parts)))

    t_part_id = t_doc.parts_ids[0]
    t_pitches = t_doc.collect_pitch_sequence(t_part_id)

    p_part_id = p_doc.parts_ids[0]
    p_pitches = p_doc.collect_pitch_sequence(p_part_id)

    edits = pitch_sequence_edits(t_pitches, p_pitches)
    print('Edits between true and pred: {0}'.format(edits))
    print('Edit distance: {0}'.format(len(edits)))

    _end_time = time.clock()
    logging.info('pitch_counter.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

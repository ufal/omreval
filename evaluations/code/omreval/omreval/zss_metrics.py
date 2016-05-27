"""This module implements classes used to measure MusicXML tree edit
distances using the lxml module."""
from __future__ import print_function

import Levenshtein

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class ZSSMetricClass(object):
    """Base class for providing costs/metrics to the ZSS tree edit distance
    module."""
    @staticmethod
    def get_children(e):
        raise NotImplementedError()

    @staticmethod
    def update(e, f):
        raise NotImplementedError()

    @staticmethod
    def insert(e):
        raise NotImplementedError()

    @staticmethod
    def remove(e):
        raise NotImplementedError()


class Lxml4ZSS(ZSSMetricClass):
    """A class that defines how edit operation costs should
    be computed from ``lxml.etree.Element`` nodes. Pass to
    ``zss.distance()``"""
    @staticmethod
    def get_children(e):
        return e.getchildren()

    @staticmethod
    def update(e, f):
        tag_equal = False
        if e.tag == f.tag:
            tag_equal = True

        text_equal = False
        if e.text is None:
            if f.text is None:
                text_equal = True
        else:
            if f.text is not None:
                if e.text.strip() == f.text.strip():
                    text_equal = True

        if (not tag_equal) or (not text_equal):
            return 1
        else:
            return 0

    @staticmethod
    def insert(e):
        return 1

    @staticmethod
    def remove(e):
        return 1


class Lxml4ZSS_Filtered(Lxml4ZSS):
    """Filters out MusicXML entities irrelevant to evaluation."""
    filtered_out = {
        'work',
        'identification',
        'defaults',
        'credit',
        'print',
        'midi-instrument',
        'midi-device',
        'duration',
    }

    @staticmethod
    def get_children(e):
        ch = e.getchildren()
        return [e for e in ch if e.tag not in Lxml4ZSS_Filtered.filtered_out]


class Lxml4ZSS_Levenshtein(Lxml4ZSS_Filtered):
    """Tries to bypass some problems inherent to the edit
    distance on nodes, when single notes are rather complex
    subtrees and deleting a rest, which can be accomplished in
    a single keystroke, costs something like six deletions."""

    @staticmethod
    def update(e, f):
        tag_change_cost = 0
        if e.tag != f.tag:
            tag_change_cost = 1

        text_edit_cost = 0
        if (e.tag == 'note') or (f.tag == 'note'):
            if e.text is None:
                text_edit_cost = len(f.text)
            elif f.text is None:
                text_edit_cost = len(e.text)
            else:
                text_edit_cost = Levenshtein.distance(e.text, f.text)
        elif e.text != f.text:
            text_edit_cost += 1

        return tag_change_cost + text_edit_cost

    @staticmethod
    def insert(e):
        if e.tag == 'note':
            return 1 + len(e.text)
        else:
            return 1

    @staticmethod
    def remove(e):
        return 1

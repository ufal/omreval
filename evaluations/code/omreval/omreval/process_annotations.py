#!/usr/bin/env python
"""This script processes annotation results for pairwise ranking."""
from __future__ import print_function, unicode_literals
import argparse
import codecs
import collections
import copy
import logging
import operator
import os
import random
import time

import cv2
import math

import itertools
import matplotlib.pyplot as plt
import numpy
import re
import scipy.stats
import scipy.misc

# import pydot

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


DEFAULT_DIRECTORY_LISTING = {
    'note_': 'single-note',
    'mozart_': 'full-fragment',
    '1-single-staff-single-voice_': 'complex',
    '2-single-staff-multi-voice_': 'complex',
    '3-multi-staff-single-voice_': 'complex',
    'scale_': 'note-sequence',
     'two-part_': 'multi-part',
    'two-part-longer_': 'multi-part',
}


def read_annotations(f):
    """Loads the annotations into the following data structure:

    Top-level keys: true,
    Value: dict with keys:
        (pred1, pred2),
    Values:
        list of (rank, annotator) pairs
    """
    if isinstance(f, str) or isinstance(f, unicode):
        handle = codecs.open(f, 'r', 'utf-8')
    else:
        handle = f

    annots = {}
    for i, line in enumerate(handle):
        try:
            true, pred1, pred2, rank, user = line.strip().split('\t')
        except ValueError:
            raise ValueError('Incorrect input format at line {0}: {1}'.format(i, line))

        if true not in annots:
            annots[true] = {}
        if (pred1, pred2) not in annots[true]:
            annots[true][(pred1, pred2)] = []
        annots[true][(pred1, pred2)].append((int(rank), user))

    if isinstance(f, str):
        handle.close()

    return annots


def item_wise_averages(annots):
    itemwise_agreement_annots = {}
    for a in annots:
        itemwise_agreement_annots[a] = {}
        for e in annots[a]:
            ranks = numpy.array(map(operator.itemgetter(0), annots[a][e]))
            itemwise_agreement_annots[a][e] = numpy.average(ranks)
    return itemwise_agreement_annots


def collect_per_annotator(annots):
    """Stores (rank, user) pairs in leaf, although "user" is top-level key,
    so that the data structure is intact."""
    per_annotator = {}
    for a in annots:
        for e in annots[a]:
            rankings = annots[a][e]
            for r, n in rankings:
                if n not in per_annotator:
                    per_annotator[n] = {}
                if a not in per_annotator[n]:
                    per_annotator[n][a] = {}
                if e not in per_annotator[n][a]:
                    per_annotator[n][a][e] = []
                per_annotator[n][a][e].append((r, n))
    return per_annotator


def reconstruct_from_per_annotator(per_annotator_annots, names=[], exclude=True):
    """Reconstructs the collection from per-annotator factorization.
    Optionally specify only some annotators and whether to include or
    exclude them"""
    annots = {}
    for n in per_annotator_annots:
        # Should we count this user?
        if n in names:
            if exclude:
                continue
        else:
            if not exclude:
                continue

        for a in per_annotator_annots[n]:
            if a not in annots:
                annots[a] = {}
            for e in per_annotator_annots[n][a]:
                if e not in annots[a]:
                    annots[a][e] = []
                for r, _ in per_annotator_annots[n][a][e]:
                    annots[a][e].append((r, n))
    return annots


def filter_annotators(annots, names=[], exclude=True):
    per_user = collect_per_annotator(annots)
    output = reconstruct_from_per_annotator(per_user, names=names, exclude=exclude)
    return output


def sum_of_absvalues_of_averages(annots):
    """Makes sense on multi-annotator annots. Sums the absolute values
    of each (true, example) pair ranking average."""
    s = 0
    for a in annots:
        for e in annots[a]:
            ranks = numpy.array(map(operator.itemgetter(0), annots[a][e]))
            avg = numpy.average(ranks)
            abs = numpy.absolute(avg)
            s += abs
    return s


def average_sum_of_absvalues_of_averages(annots):
    s = sum_of_absvalues_of_averages(annots)
    avg = s / get_n_examples(annots)
    return avg


def extract_control_group(annots, control_exp=re.compile('_true$'), exclude=False):
    """Filter annotations to only keep examples where at least one predicted
    score matches the regex."""
    filtered_annots = {}
    for a in annots:
        filtered_annots[a] = {}
        for e in annots[a]:
            if re.search(control_exp, e[0]) or re.search(control_exp, e[1]):
                if exclude:
                    continue
            else:
                if not exclude:
                    continue
            filtered_annots[a][e] = copy.deepcopy(annots[a][e])
    return filtered_annots


def get_n_rankings(annots):
    n = 0
    for a in annots:
        for e in annots[a]:
            n += len(annots[a][e])
    return n


def get_n_examples(annots):
    n = 0
    for a in annots:
        for e in annots[a]:
            n += 1
    return n


def collect_leaf_values(annots):
    v = []
    for a in annots:
        for e in annots[a]:
            v.append(annots[a][e])
    return v


def generate_adverse_annotator(annots_iwa, name='adverse'):
    """Generates the annotations by an adverse annotator: someone
    dedicated to disagreeing as much as possible."""
    adverse = {}
    for a in annots_iwa:
        adverse[a] = {}
        for e in annots_iwa[a]:
            if annots_iwa[a][e] > 0:
                adverse[a][e] = [(-1, name)]
            else:
                adverse[a][e] = [(1, name)]
    return adverse


def generate_random_annotator(annots_iwa, name='random'):
    """Generates the annotations by an adverse annotator: someone
    dedicated to disagreeing as much as possible."""
    random_annot = {}
    for a in annots_iwa:
        random_annot[a] = {}
        for e in annots_iwa[a]:
            if numpy.random.uniform(0, 1) < 0.5:
                random_annot[a][e] = [(-1, name)]
            else:
                random_annot[a][e] = [(1, name)]
    return random_annot


def split_annots(annots, groups):
    """Split the annotation by groups of annotators.

    :param groups: List of lists of names.
    """
    per_group = {}
    for g in groups:
        g_annots = filter_annotators(annots, names=g, exclude=False)
        per_group[tuple(g)] = g_annots
    return per_group


def merge_annots(*annots):
    """Combine all the annotation dicts into one."""
    all = {}
    for a in annots:
        per_user = collect_per_annotator(a)
        for n in per_user:
            if n not in all:
                all[n] = per_user[n]
            else:
                raise ValueError('merge_annots cannot deal with duplicate '
                                 'annotators! Problem: {0}'.format(n))
    return reconstruct_from_per_annotator(all, names=[], exclude=True)


def filter_annots_by_value(annots, condition_fn):
    output = {}
    for a in annots:
        output[a] = {}
        for e in annots[a]:
            if condition_fn(annots[a][e]):
                output[a][e] = annots[a][e]
    return output


def filter_annots_by_example(annots, examples, exclude=True):
    output = {}
    for a in annots:
        output[a] = {}
        for e in annots[a]:
            if e in examples:
                if exclude:
                    continue
            else:
                if not exclude:
                    continue
            output[a][e] = annots[a][e]
    return output


def filter_annots_by_true(annots, trues, exclude=True):
    if exclude:
        return {a: annots[a] for a in annots if a not in trues}
    else:
        return {a: annots[a] for a in annots if a in trues}


def flatten_annots_to_examples(annots):
    output = {}
    for a in annots:
        for e in annots[a]:
            output[e] = annots[a][e]
    return output


def rankings2preference_points(annots, as_proportion=False, invert=False):
    prefs = {}
    participation = {}
    for a in annots:
        prefs[a] = {}
        participation[a] = {}
        for e in annots[a]:
            l, r  = e
            if l not in prefs[a]:
                prefs[a][l] = 0
            if r not in prefs[a]:
                prefs[a][r] = 0
            if l not in participation[a]:
                participation[a][l] = 0.0
            if r not in participation[a]:
                participation[a][r] = 0.0

            participation[a][l] += len(annots[a][e])
            participation[a][r] += len(annots[a][e])

            for k, u in annots[a][e]:
                if k < 0:
                    if invert:
                        prefs[a][r] += 1
                    else:
                        prefs[a][l] += 1
                else:
                    if invert:
                        prefs[a][l] += 1
                    else:
                        prefs[a][r] += 1

        if as_proportion:
            for x in prefs[a]:
                prefs[a][x] /= participation[a][x]

    return prefs


def format_costs_as_badness(costs,
                            directory_listing=DEFAULT_DIRECTORY_LISTING,
                            suffix='.xml'):
    """Prints out the costs in the format expected by badness-based eval
    scripts."""
    lines = []
    for a in costs:
        a_fname = a + suffix
        for dcode, dprefix in directory_listing.iteritems():
            if a_fname.startswith(dcode):
                a_fname = os.path.join(dprefix, a_fname)
                break

        for x in costs[a]:
            # Apply directory listing: prefixes to point from database names
            # back to filenames.
            x_fname = x + suffix
            for dcode, dprefix in directory_listing.iteritems():
                if x_fname.startswith(dcode):
                    x_fname = os.path.join(dprefix, x_fname)
            lines.append('{0}\t{1}\t{2}'.format(a_fname, x_fname, costs[a][x]))

    return '\n'.join(lines) + '\n'


def format_iwa_as_rankings(iwa,
                           directory_listing=DEFAULT_DIRECTORY_LISTING,
                           suffix='.xml',
                           equality_absvalue_threshold=1./3.):
    lines = []
    for a in iwa:
        a_fname = annotname2filename(a, directory_listing, suffix)

        for e in iwa[a]:
            el, er = e
            el_fname = annotname2filename(el, directory_listing, suffix)
            er_fname = annotname2filename(er, directory_listing, suffix)
            if iwa[a][e] ** 2 < equality_absvalue_threshold ** 2:
                rank = 0
            elif iwa[a][e] > 0:
                rank = 1
            else:
                rank = -1

            lines.append('{0}\t{1}\t{2}\t{3}'.format(a_fname, el_fname, er_fname, rank))

    return '\n'.join(lines) + '\n'


def annotname2filename(a, directory_listing, suffix):
    a_fname = a + suffix
    for dcode, dprefix in directory_listing.iteritems():
        if a_fname.startswith(dcode):
            a_fname = os.path.join(dprefix, a_fname)
            break
    return a_fname


def annots2annotnames_and_filelist(annots, directory_listing, suffix):
    """Returns a dictionary of annot name --> filename (relative to root)"""
    annot_names = {}
    for a in annots:
        annot_names[a] = 1
        for e1, e2 in annots[a]:
            annot_names[e1] = 1
            annot_names[e2] = 1
    annot_names = sorted(annot_names.keys())
    filelist = [annotname2filename(a, directory_listing=directory_listing, suffix=suffix) for a in annot_names]
    return dict(zip(annot_names, filelist))


def format_annots_as_cost_pair_filenames(annots):
    a2f = annots2annotnames_and_filelist(annots, directory_listing=DEFAULT_DIRECTORY_LISTING, suffix='.xml')
    pairs = set()
    for a in annots:
        for e1, e2 in annots[a]:
            if (a, e1) not in pairs:
                pairs.add((a, e1))
            if (a, e2) not in pairs:
                pairs.add((a, e2))
    lines = ['{0}\t{1}'.format(a2f[a], a2f[e]) for a, e in sorted(pairs)]
    return '\n'.join(lines)





def annots_as_edges(annots, true_names=[], exclude=True):
    """Edges lead from better sysoutput to worse."""
    edges = []
    for a in annots:
        if a in true_names:
            if exclude:
                continue
        else:
            if not exclude:
                continue

        for e in annots[a]:
            for r, u in annots[a][e]:
                el, er = e
                if r > 0:
                    edges.append((er, el))
                elif r < 0:
                    edges.append((el, er))

    return edges


# def annotation_to_graph(annots, **edge_kwargs):
#     """Draws the annotations as a graph."""
#     edges = annots_as_edges(annots, **edge_kwargs)
#     g = pydot.graph_from_edges(edges, directed=True)
#     return g
#
#
# def draw_annotation_graph(annots, output_file, **edge_kwargs):
#     g = annotation_to_graph(annots, **edge_kwargs)
#     g.write(output_file, format='png')
#
#
# def show_annot_graph(annots, temp_output_file='temp_output_file.annot_processing.py.png',
#                      **edge_kwargs):
#     g = annotation_to_graph(annots, **edge_kwargs)
#     g.write(temp_output_file, format='png')
#     img = cv2.imread(temp_output_file)
#     os.remove(temp_output_file)
#     plt.imshow(img)
#     plt.show()


def annots_to_badness(input_file, output_file):
    """Converts annotation results into a badness-based test case file.
    End-to-end method for processing annotations.

    To get badness (cost), for each predicted score we count how many
    times it lost in a comparison, relative to the total number of
    comparisons it participated in.
    """
    with codecs.open(input_file, 'r', 'utf-8') as input_handle:
        annots = read_annotations(input_handle)

    costs = rankings2preference_points(annots, as_proportion=False, invert=True)
    s = format_costs_as_badness(costs)

    with codecs.open(output_file, 'w', 'utf-8') as output_handle:
        output_handle.write(s)


def annots_to_rankings(input_file, output_file):
    """Converts annotation results into a ranking-based test case file.
    End-to-end method for processing annotations.

    To get the output ranking for an item: compute the average annotator
    ranking; if it is 1/3 or less, consider it equal.
    """
    with codecs.open(input_file, 'r', 'utf-8') as input_handle:
        annots = read_annotations(input_handle)

    iwa = item_wise_averages(annots)
    s = format_iwa_as_rankings(iwa)

    with codecs.open(output_file, 'w', 'utf-8') as output_handle:
        output_handle.write(s)


def iwa_mse(iwa1, iwa2):
    """Measures only the intersection of iwa1 and iwa2."""
    pk = []
    qk = []
    for a in iwa1:
        if a not in iwa2:
            continue
        a1 = iwa1[a]
        a2 = iwa2[a]
        for e in a1:
            if e not in a2:
                continue
            i1 = a1[e]
            i2 = a2[e]
            pk.append(i1)
            qk.append(i2)

    pk = numpy.array(pk)
    qk = numpy.array(qk)
    mean_squared_error = numpy.average((pk - qk) ** 2)

    return mean_squared_error


def iwa_squared_error_dict(iwa1, iwa2):
    d = {}
    for a in iwa1:
        if a not in iwa2:
            continue
        d[a] = {}
        a1 = iwa1[a]
        a2 = iwa2[a]
        for e in a1:
            if e not in a2:
                continue
            d[a][e] = (a1[e] - a2[e]) ** 2

    return d


def mse_of_annotator_groups(annots, names1, names2=None):
    per_user = collect_per_annotator(annots)
    annots1 = reconstruct_from_per_annotator(per_user, names=names1, exclude=False)
    if names2:
        annots2 = reconstruct_from_per_annotator(per_user, names=names2, exclude=False)
    else:
        annots2 = reconstruct_from_per_annotator(per_user, names=names1, exclude=True)

    iwa1 = item_wise_averages(annots1)
    iwa2 = item_wise_averages(annots2)
    mse = iwa_mse(iwa1, iwa2)
    return mse


def random_names(names, k):
    """Return randomly chosen K names."""
    if k > len(names):
        raise ValueError("Trying to choose {0} from a set of {1} names!"
                         "".format(k, len(names)))
    nnames = copy.deepcopy(names)
    output = []
    for i in xrange(k):
        n = random.choice(nnames)
        output.append(n)
        nnames = [m for m in nnames if m not in output]
    return output


def pairwise_agreement(n1, n2, annots,
                       weighed=False, exclude_from_consensus=[]):
    """Compute the proportion of test cases on which annotators `n1` and `n2`
    agree.

    If `weighed`, will weigh the individual test cases by the extent
    to which the other annotators agree: disagreement on items where consensus
    is strong will thus hurt overall agreement more than disagreement on items
    where the others do not agree with each other as well.

    Weighing only deals with consensus strength, not whether the annotators
    in question agree with the consensus or not.
    """
    per_user = collect_per_annotator(annots)
    paired = reconstruct_from_per_annotator(per_user, names=[n1, n2], exclude=False)
    paired_iwa = item_wise_averages(paired)
    # The values in paired_iwa are -1 (for both saying -1), 0 (for disagreement),
    # or 1 (for both saying +1).

    if not weighed:
        iwas = numpy.array(collect_leaf_values(paired_iwa))

    else:
        others = reconstruct_from_per_annotator(per_user,
                                                names=[n1, n2] + exclude_from_consensus,
                                                exclude=True)
        others_iwa = item_wise_averages(others)
        others_examples_iwa = flatten_annots_to_examples(others_iwa)
        paired_examples_iwa = flatten_annots_to_examples(paired_iwa)
        weighed_examples_iwa = {e: others_examples_iwa[e] * x
                                for e, x in paired_examples_iwa.items()
                                if e in others_examples_iwa}
        iwas = numpy.array(weighed_examples_iwa.values())

    abs_iwas = numpy.absolute(iwas)
    avg_agreement = numpy.average(abs_iwas)
    return avg_agreement


def pairwise_agreement_profile(n, annots,
                               weighed=False,
                               exclude_from_consensus=[]):
    per_user = collect_per_annotator(annots)

    agreements = {}
    proportional_agreements = {}
    maxima = {}

    for m in per_user.keys():
        if m == n:
            continue
        else:
            a = pairwise_agreement(n, m, annots, weighed=weighed,
                                   exclude_from_consensus=exclude_from_consensus)

            maximum = pairwise_agreement(n, n, annots, weighed=weighed,
                                         exclude_from_consensus=[m] + exclude_from_consensus)

            agreements[m] = a
            maxima[m] = maximum
            proportional_agreements[m] = a / maximum

    return maxima, agreements, proportional_agreements


def pairwise_agreement_profiles2matrix(agreement_profile_per_user):
    ns = sorted(agreement_profile_per_user.keys())
    prof_matrix = numpy.zeros((len(ns), len(ns)), dtype='float64')
    for i, n in enumerate(ns):
        for j, m in enumerate(ns):
            if m == n:
                prof_matrix[i][j] = 1.0
            else:
                prof_matrix[i][j] = agreement_profile_per_user[n][2][m]
    return prof_matrix


def annots2pairwise_agreement_profile_matrix(annots,
                                             weighed=False, **pairwise_kwargs):
    per_user = collect_per_annotator(annots)
    ns = sorted(per_user.keys())
    agreement_profiles = {n: pairwise_agreement_profile(n, annots, weighed=weighed,
                                                        **pairwise_kwargs)
                          for n in ns}
    matrix = pairwise_agreement_profiles2matrix(agreement_profiles)
    return matrix


def plot_pairwise_agreement_matrices(annots, **pairwise_kwargs):
    u_mat = annots2pairwise_agreement_profile_matrix(annots, weighed=False,
                                                     **pairwise_kwargs)
    w_mat = annots2pairwise_agreement_profile_matrix(annots, weighed=True,
                                                     **pairwise_kwargs)

    per_user = collect_per_annotator(annots)
    ns = sorted(per_user.keys())

    vmax = max(u_mat.max(), u_mat.max())
    vmin = min(u_mat.min(), u_mat.min())

    plt.subplot(1, 2, 1)
    plt.imshow(u_mat, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(len(ns)), ns, rotation='vertical')
    plt.yticks(range(len(ns)), ns)


    plt.subplot(1, 2, 2)
    plt.imshow(w_mat, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(len(ns)), ns, rotation='vertical')
    plt.yticks(range(len(ns)), ns)


def plot_weighed_pairwise_agreement_matrix(annots, w_mat=None,
                                           vmin=None, vmax=1.0,
                                           limit_to_first_n=None,
                                           names=True,
                                           **pairwise_kwargs):
    if w_mat is None:
        w_mat = annots2pairwise_agreement_profile_matrix(annots, weighed=True,
                                                         **pairwise_kwargs)

    vmax = max(w_mat.max(), w_mat.max())
    if vmin is None:
        vmin = min(w_mat.min(), w_mat.min())

    if limit_to_first_n is None:
        limit_to_first_n = w_mat.shape[0]

    per_user = collect_per_annotator(annots)
    ns = sorted(per_user.keys())[:limit_to_first_n]

    plt.imshow(w_mat[:limit_to_first_n, :limit_to_first_n],
               interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if names:
        plt.xticks(range(len(ns)), ns, rotation='vertical')
        plt.yticks(range(len(ns)), ns)
    else:
        plt.xticks(range(len(ns)), range(len(ns)))
        plt.yticks(range(len(ns)), range(len(ns)))


def update_with_random(annots, n_random):
    random_names = ['zz_random_{0}'.format(i) for i in xrange(n_random)]
    iwa = item_wise_averages(annots)
    rs = {n: generate_random_annotator(iwa, n) for n in random_names}
    per_user = collect_per_annotator(annots)
    for n in rs:
        per_user[n] = rs[n]
    output = reconstruct_from_per_annotator(per_user)
    return output, random_names


def disagreement_signature(annots, name):
    """For each annotation done by name, collect names of annotators
    that disagreed. Output is organized by example."""
    output = {}
    for a in annots:
        for e in annots[a]:
            names = map(operator.itemgetter(1), annots[a][e])
            ranks = map(operator.itemgetter(0), annots[a][e])
            nr_d = {n: r for n, r in zip(names, ranks)}
            if name in nr_d:
                r_name = nr_d[name]
                disagreed = [n for n in nr_d if nr_d[n] != r_name]
                output[e] = disagreed
    return output


def example_consensus_weights(annots, names=[], exclude=True):
    """Get consensus-based weight per example."""
    cons_annots = filter_annotators(annots, names=names, exclude=exclude)
    iwa = item_wise_averages(cons_annots)
    example_iwa = flatten_annots_to_examples(iwa)
    abs_example_iwa = {e: numpy.absolute(example_iwa[e]) for e in example_iwa}
    return abs_example_iwa


def sort_user_disagreement_examples_by_weight(n, annots):
    weights = example_consensus_weights(annots, names=[n], exclude=True)
    consensus_iwa = flatten_annots_to_examples(
        item_wise_averages(
            filter_annotators(annots, names=[n], exclude=True)
        )
    )

    user_annots = collect_per_annotator(annots)[n]
    examples = flatten_annots_to_examples(user_annots)
    sorted_examples = sorted([e for e in examples.items() if e[0] in weights],
                             key=lambda x: weights[x[0]], reverse=True)
    disagreements = [(e, r) for e, r in sorted_examples if r * consensus_iwa[e] < 0]
    return disagreements


def n_disagreements_in_annots(annots):
    """Counts the number of conflicts in the annotation."""
    n_d = 0
    for a in annots:
        for e in annots[a]:
            r, u = zip(*annots[a][e])
            n_ones = len([a for a in r if a > 0])
            n_conflicts = n_ones * (len(r) - n_ones)
            n_d += n_conflicts
    return n_d


def disagreement_by_iwa_bins(annots, n_bins=None, bin_bounds=None):
    """Counts how many disagreements are there per certainty
    level."""
    if bin_bounds is None:
        if n_bins is None:
            K = len(collect_per_annotator(annots))
            n_bins = K / 2 + (K % 2) + 1

        bin_bounds = numpy.arange(0, 1.0 + 0.0001, 1.0 / (n_bins - 1))
    else:
        n_bins = len(bin_bounds)

    def get_bin(x):
        for i, b in enumerate(bin_bounds):
            if x <= b:
                return i
        raise ValueError('Too high x={0} for bin bounds {1}'.format(x, bin_bounds))

    def get_n_disagreements_in_item(item):
        r, u = zip(*item)
        n_ones = len([a for a in r if a > 0])
        n_conflicts = n_ones * (len(r) - n_ones)
        return n_conflicts

    bins = [0 for _ in xrange(n_bins)]
    iwa = item_wise_averages(annots)
    for a in annots:
        for e in annots[a]:
            bin_id = get_bin(numpy.sqrt(iwa[a][e] ** 2) - 0.00001)
            n_disagreements = get_n_disagreements_in_item(annots[a][e])
            bins[bin_id] += n_disagreements

    return bins, bin_bounds


def iwa_bins(annots, n_bins=None, bin_bounds=None):
    """Counts how many items fall into which disagreement level."""
    if bin_bounds is None:
        if n_bins is None:
            K = len(collect_per_annotator(annots))
            n_bins = K / 2 + (K % 2) + 1

        bin_bounds = numpy.arange(0, 1.0 + 0.0001, 1.0 / (n_bins - 1))
    else:
        n_bins = len(bin_bounds)

    def get_bin(x):
        for i, b in enumerate(bin_bounds):
            if x <= b:
                return i
        raise ValueError('Too high x={0} for bin bounds {1}'.format(x, bin_bounds))

    bins = [0 for _ in xrange(n_bins)]
    iwa = item_wise_averages(annots)
    for a in iwa:
        for e in iwa[a]:
            bin_id = get_bin(numpy.sqrt(iwa[a][e] ** 2) - 0.00001)
            bins[bin_id] += 1
    return bins, bin_bounds


################################
#
# Probabilistic modeling of the annotations
#
def p_Tc_is_0_given_rc(K, n_rc_1):
    return scipy.misc.comb(K, n_rc_1, exact=True) / (2.0 ** K)


def p_rC_given_TC_qA(rC, TC, qA):
    """
    :param rC: example dict, values are lists of (rank, user) tuples
    :param tC: examples dict, keys are consensus assignment (-1, 0, 1)
               probs: 3-member array P(tC = 0), P(tC = 1), P(tC = -1)
               (so that it is indexable directly by ranking...)
    :param qA: annotator error dict, key is (user), values are q_a
    :return: P(r(C) | T(C), q_A)
    """
    p = 1.0
    for e, rc in rC.items():
        tc = TC[e]
        p_c = 1.0
        for rac, a in rc:
            qa = qA[a]
            p_rac = 0.5 * tc[0] + (1 - qa) * tc[rac] + qa * tc[rac * 2]
            p_c *= p_rac
        p *= p_c
    return p


def p_rC_given_TC_q(rC, TC, q, K):
    """The inner conditional for \tilde{r}(C). Binomial distributions
    on the number of +1 votes, according to TC probs.

    :param rC: example dict, values are lists of (rank, user) tuples
    :param tC: examples dict, keys are consensus assignment (-1, 0, 1)
               probs: 3-member array P(tC = 0), P(tC = 1), P(tC = -1)
               (so that it is indexable directly by ranking...)
    :param q:  expected annotator error q
    :param K:  Total number of annotators

    :return: P(r(C) | T(C), q_A)
    """
    p = 1.0
    # Precompute binomial distribution table
    binom_table = [[scipy.stats.binom.pmf(n, K, 0.5),
                    scipy.stats.binom.pmf(n, K, 1.0 - q),
                    scipy.stats.binom.pmf(n, K, q)] for n in xrange(K+1)]

    for e, rc in rC.items():
        tc = TC[e]
        n_rc_1 = len([rac for rac, a in rc if rac == 1])
        p_c = binom_table[n_rc_1][0] * tc[0] + \
              binom_table[n_rc_1][1] * tc[1] + \
              binom_table[n_rc_1][-1] * tc[-1]
        p *= p_c
    return p


def p_rC_given_qA(rC, qA, iwa, K, eps=0.0001,
                  simplified=True,
                  n_randomize_TCs=0,
                  TC_0_pertrubation=-0.0001,
                  TC_non0_pertrubation=-0.0001):
    """Marginalizes over all T(C) such that assignments T(c)
    are consistent with r(c). Still needs a given qA.

    You can also specify how many randomized TCs should be generated
    and a probability of pertrubing a zero-value T(c) towards consensus.
    """
    TC_templates = {0: [1.0 - 2*eps, eps, eps],
                    1: [eps, 1.0 - 2*eps, eps],
                    -1: [eps, eps, 1.0 - 2*eps]}

    # Collect iwa values
    iwa_abs_values = sorted(set(numpy.absolute(numpy.array(collect_leaf_values(iwa)))))
    example_iwa = flatten_annots_to_examples(iwa)

    # To avoid floating-point trouble
    _TC_eps = 0.0001

    p_rC_given_qA_TC = []
    if not n_randomize_TCs:
        for TC_lim in iwa_abs_values:
            # - Generate TC assignment,
            TC = {}
            for e, i in example_iwa.iteritems():
                if numpy.absolute(i) <= TC_lim + _TC_eps:
                    TC[e] = TC_templates[0]
                    if numpy.random.uniform(0, 1.0) > TC_0_pertrubation:
                        # Don't pertrube, keep 0
                        continue
                if i < 0:
                    if numpy.random.uniform(0, 1.0) < TC_non0_pertrubation:
                        TC[e] = TC_templates[0]
                    else:
                        TC[e] = TC_templates[-1]
                else:
                    if numpy.random.uniform(0, 1.0) < TC_non0_pertrubation:
                        TC[e] = TC_templates[0]
                    else:
                        TC[e] = TC_templates[1]

            # - Compute P(r(C) | T(C), q_A)
            if simplified:
                q = numpy.average(numpy.array(qA.values()))
                p = p_rC_given_TC_q(rC, TC, q, K)
            else:
                p = p_rC_given_TC_qA(rC, TC, qA)
            p_rC_given_qA_TC.append(p)

    # Random TCs... these contribute very little
    else:
        for _ in xrange(n_randomize_TCs):
            TC = {
                e: TC_templates[random.choice([-1, 0, 1])] for e in example_iwa
            }
            if simplified:
                q = numpy.average(numpy.array(qA.values()))
                p = p_rC_given_TC_q(rC, TC, q, K)
            else:
                p = p_rC_given_TC_qA(rC, TC, qA)
            p_rC_given_qA_TC.append(p)

    return p_rC_given_qA_TC


def p_rC(rC, iwa, names, qs=numpy.arange(0.001, 0.5, 0.01), eps=0.0001,
         **kwargs):

    output = numpy.array([p_rC_given_qA(rC,
                                        {n: q for n in names},
                                        iwa,
                                        len(names),
                                        **kwargs)
                          for q in qs])
    return output


def annots2rC(annots):
    return flatten_annots_to_examples(annots)


def annots2R(annots):
    """Convert annotations into a numpy array representing
    the R variables. Annotators are columns (sorted by name),
    examples are rows (also sorted by example name)."""
    flat = flatten_annots_to_examples(annots)
    n_annotators = len(flat.items()[0][1])
    n_examples = len(flat)
    R = numpy.zeros((n_examples, n_annotators))
    examples = sorted(flat.items(), key=operator.itemgetter(0))
    for i in xrange(n_examples):
        ex = examples[i]
        rs = sorted(ex[1], key=operator.itemgetter(1))
        for k, a in enumerate(rs):
            R[i, k] = a[0]
    return R


def annots2TC(annots, xi_0=0.05, eps=0.0001):
    """Prepare consensus assignment.
    So far: hard assignment.

    :param annots: The annotations from which to derive consensus T(C).
    :param xi_0: Assuming P(r_a(c) = 0.5 | T(c) = 0), the threshold P(T(c) = 0 | r(c))
                 over which an item is assigned T(c) = 0. Higher settings mean
                 less propensity to label items as equal.
    :param eps: Epsilon given to T(c) that would otherwise be zero.
    """
    iwa = item_wise_averages(annots)
    example_iwa = flatten_annots_to_examples(iwa)

    # Find T(c) = 0 threshold expressed in absolute value of item-wise agreement
    per_user = collect_per_annotator(annots)
    K = len(per_user)
    Tc_is_0_iwa_upper_threshold = 0.0
    for n_rc_1 in xrange(K / 2 + (K % 2) + 1):
        if p_Tc_is_0_given_rc(K, n_rc_1) > xi_0:
            Tc_is_0_iwa_upper_threshold = (1.0 / K) * n_rc_1 + 0.00001
            # print('Threshold for Tc = 0: {0}, n_rc = {1}'
            #       ''.format(Tc_is_0_iwa_upper_threshold, n_rc_1))
            break

    TC_templates = {0: [1.0 - 2*eps, eps, eps],
                    1: [eps, 1.0 - 2*eps, eps],
                    -1: [eps, eps, 1.0 - 2*eps]}
    TC = {}
    for e in example_iwa:
        if numpy.absolute(example_iwa[e]) < Tc_is_0_iwa_upper_threshold:
            TC[e] = TC_templates[0]
        else:
            if example_iwa[e] < 0:
                TC[e] = TC_templates[-1]
            else:
                TC[e] = TC_templates[1]

    return TC


def annots2T(annots, xi_0=0.05, eps=0.0001):
    """Prepare consensus assignment for inference.

    :return: A numpy array of shape (n_examples, 3) where
        each row has prob. of T(c) being 0, +1 and -1 (in this
        order, so that T[i][-1] is prob. of -1).
    """
    Tc = annots2TC(annots, xi_0=xi_0, eps=eps)
    T = numpy.zeros((len(Tc), 3))
    for i, (e, f) in enumerate(sorted(Tc.items(), key=operator.itemgetter(0))):
        T[i] = f
    return T


def annots2TC_soft(annots, qA, xi_0=0.05, eps=0.0001):
    """Prepare soft consensus assignment.

    Soft assignment is based on the IWA of an item: instead of just


    :param annots: The annotations from which to derive consensus T(C).
    :param xi_0: Assuming P(r_a(c) = 0.5 | T(c) = 0), the threshold P(T(c) = 0 | r(c))
                 over which an item is assigned T(c) = 0. Higher settings mean
                 less propensity to label items as equal.
    :param eps: Epsilon given to T(c) that would otherwise be zero.
    """
    raise NotImplementedError()


def inference(R, T, Q, eps=0.0001, max_iterations=10):
    """Infer T(C), Q based on R. The model has separate q's per annotator.

    :param R: A numpy array of shape (nb_examples, nb_annotators)
        with values -1, +1.

    :param T: A numpy array of shape (nb_examples, 3) with values between
        0 and 1 s. t. each row sums to 1.

    :param Q: A numpy array of shape (nb_annotators, ) with values
        between 0 and 1.

    :return: T, Q: The inferred assignment of T(C) and Q_a parameters.
    """
    # Check shapes
    if R.shape[0] != T.shape[0]:
        raise ValueError('Wrong shapes: R.shape = {0}, T.shape = {1}'
                         ''.format(R.shape, T.shape))
    if R.shape[1] != Q.shape[0]:
        raise ValueError('Wrong shapes: R.shape = {0}, Q.shape = {1}'
                         ''.format(R.shape, Q.shape))

    TC_templates = {0: numpy.array([1.0 - 2*eps, eps, eps]),
                    1: numpy.array([eps, 1.0 - 2*eps, eps]),
                    -1: numpy.array([eps, eps, 1.0 - 2*eps])}

    def get_item_cost(Tc, Rc, Q):
        cost_tie = 0.5 ** Q.shape[0]
        cost_plus = numpy.prod(1.0 - Q[Rc > 0]) * numpy.prod(Q[Rc < 0])
        cost_minus = numpy.prod(1.0 - Q[Rc < 0]) * numpy.prod(Q[Rc > 0])
        return numpy.array([cost_tie, cost_plus, cost_minus])

    def sample_Tc_from_costs(c_costs):
        normalized_costs = c_costs / c_costs.sum()
        cost_bins = [normalized_costs[:k+1].sum() for k in xrange(3)]
        rand = numpy.random.uniform(0, 1)
        # Find the bin which was hit
        b = 0
        if rand > cost_bins[0]:
            b = 1
        if rand > cost_bins[1]:
            b = -1
        return b

    q_delta_threshold = 0.001
    max_q_delta = 1.0
    iteration = 0
    while (max_q_delta > q_delta_threshold) and (iteration < max_iterations):
        logging.debug('Iteration {0}: max_q_delta = {1}'.format(iteration, max_q_delta))
        n_Tc_changes = 0
        # Sample T(C)
        logging.debug('Sampling T(C)...')
        for i in xrange(T.shape[0]):
            # Current cost:
            Tc = T[i]
            Rc = R[i]
            c_costs = get_item_cost(Tc, Rc, Q)
            new_Tc_bin = sample_Tc_from_costs(c_costs)
            # print('\tItem {0}\n\t\tOrig:\t{1}\n\t\tCost:\t{2}\n\t\tBin:\t{3}'
            #       ''.format(i, Tc, c_costs, new_Tc_bin))
            # Logging changes
            if (TC_templates[new_Tc_bin] - Tc).sum() > 0.001:
                n_Tc_changes += 1
            T[i] = TC_templates[new_Tc_bin]
        # EM for q
        logging.debug('\tTotal changes: {0}'.format(n_Tc_changes))
        logging.debug('EM steps for Q')
        max_q_delta = 0.0
        total_q_delta = 0.0
        for a in xrange(Q.shape[0]):
            # E-step
            hits, misses, ties = 0.0, 0.0, 0.0
            for i in xrange(T.shape[0]):
                ties += T[i][0]
                hits += T[i][R[i, a]]
                misses += T[i][R[i, a] * -1]
            logging.debug('Annotator {0}: h/m/t {1:.1f}/{2:.1f}/{3:.1f}'.format(a, hits, misses, ties))
            # M-step
            new_q = misses / (hits + misses)
            # Checking for threshold...
            q_a_delta = numpy.sqrt((Q[a] - new_q) ** 2)
            total_q_delta += q_a_delta
            if q_a_delta > max_q_delta:
                max_q_delta = q_a_delta
            Q[a] = new_q
        logging.debug('\tTotal delta: {0}'.format(total_q_delta))
        iteration += 1

    return T, Q

##############################################################################

# Displaying the images
TEST_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data')
IMAGE_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data-images')


def name2file(name,
              root=TEST_DATA_ROOT,
              directory_listing=DEFAULT_DIRECTORY_LISTING,
              suffix='.xml'):
    fname = name + suffix
    for dcode, dprefix in directory_listing.iteritems():
        if fname.startswith(dcode):
            fname = os.path.join(dprefix, fname)
            break
    return os.path.join(root, fname)


def file2name(filename):
    basename = os.path.basename(filename)
    no_suffix = basename.rsplit('.', 1)[0]
    return no_suffix


def name2image_file(name):
    return name2file(name,
                     root=IMAGE_DATA_ROOT,
                     directory_listing=DEFAULT_DIRECTORY_LISTING,
                     suffix='.png')


def plot_image(name, show=False):
    fname = name2image_file(name)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    plt.imshow(img, interpolation='nearest')
    if show:
        plt.show()

##############################################################################
##############################################################################
##############################################################################


# Cost processing
def read_costs(filename):
    """Parses a costs file:

        true [\t] system output [\t] cost

    The costs file is a file generated by an automated OMR evaluation
    metric (such as the **c14n**, **TED**, **TEDn** or *Ly* metrics
    from the article "Further Steps Towards a Standard Testbed for OMR")
    that assigns to each system output some cost (=how badly is the system
    output mangled with respect to the true score it was derived from).
    Cost files for the metrics implemented in ``omreval`` are in the
    ``evaluations/costs/`` directory of the article's supplementary
    materials.
    """
    costs = {}
    with codecs.open(filename, 'r', 'utf-8') as h:
        for line in h:
            if line.strip() == '':
                continue
            try:
                tf, pf, c = line.strip().split()
            except ValueError:
                raise ValueError('Wrong line format: {0}'.format(line))
            t = file2name(tf)
            p = file2name(pf)
            if t not in costs:
                costs[t] = {}
            costs[t][p] = float(c)
    return costs


def cost_diffs_for_annots(costs, annots):
    """Negative cost diff means preference for first system output
    in example, positive cost diff means preference for second."""
    cost_diffs = {}
    for a in annots:    # a = 'true score group' key
        cost_diffs[a] = {}
        for e in annots[a]:   # e = example (a tuple of system outputs)
            cost_diff = costs[a][e[0]] - costs[a][e[1]]
            cost_diffs[a][e] = cost_diff
    return cost_diffs


def classify_cost_diffs(cost_diffs, threshold):
    """From cost diffs to hard-coding the cost function's preferences.
    The threshold controls the maximum cost difference that is considered
    equal."""
    cost_diff_clf = {}
    for a in cost_diffs:
        cost_diff_clf[a] = {}
        for e, c in cost_diffs[a].items():
            if numpy.abs(c) < threshold:
                cost_diff_clf[a][e] = 0
            else:
                cost_diff_clf[a][e] = math.copysign(1, c)
    return cost_diff_clf


def cost_score(cost_diff_classification, iwa, hard=False):
    """Hard: each agreeing sign is +1, soft directly copies IWA.
    Requires all values to also be in IWA.

    This wasn't used in the article.
    """
    score = 0
    for a in cost_diff_classification:
        if a not in iwa:
            raise ValueError('IWAs for ideal {0} not found! Available: {1}'.format(a, iwa.keys()))
        for e, t in cost_diff_classification[a].items():
            if e not in iwa[a]:
                raise ValueError('IWA of example {0} for ideal {1} not found!'.format(e, a))
            if (t * iwa[a][e]) > 0:
                if hard:
                    score += 1
                else:
                    score += iwa[a][e]
    return score


def cost_scores_for_thresholds(cost_diffs, thresholds, iwa, hard=False):
    output = {t: cost_score(classify_cost_diffs(cost_diffs, t), iwa, hard=hard)
              for t in thresholds}
    return output


def cost_diffs_vs_iwas(cost_diffs, iwa):
    """Return a (diffs, iwas) tuple of numpy arrays of the cost diff
    and corresponding IWA (Item-Wise Averages, annotator agreement stats
    for the test cases)."""
    d = []
    i = []
    for a in cost_diffs:
        for e in cost_diffs[a]:
            d.append(cost_diffs[a][e])
            i.append(iwa[a][e])
    return numpy.array(d), numpy.array(i)


def costs2metrics(costs, iwa, expected_upper_bounds=None):
    """This is the function that ties together costs and annotator
    agreement and outputs the correlation coefficients for the given
    cost function and given set of human judgments.

    :returns: spearman r, pearson r, kendall tau
        If `expected_upper_bounds` is given, it will return
        the ratio of each correlation coefficient to the given
        upper bound.
    """
    cost_diffs = cost_diffs_for_annots(costs, iwa)
    cds, iwas = cost_diffs_vs_iwas(cost_diffs, iwa)
    s = scipy.stats.spearmanr(cds, iwas).correlation
    p = scipy.stats.pearsonr(cds, iwas)[0]
    k = scipy.stats.kendalltau(cds, iwas).correlation

    if expected_upper_bounds is not None:
        s_hat = s / expected_upper_bounds[0]
        p_hat = p / expected_upper_bounds[1]
        k_hat = k / expected_upper_bounds[2]

        return s, s_hat, p, p_hat, k, k_hat

    return s, p, k


def cost_file_report(filename, iwa, expected_upper_bounds=None):
    """Utility function for reporting a cost function's performance,
    as measured by the correlation coefficients, directly formatted
    as a LaTeX table line."""
    costs = read_costs(filename)
    metrics = costs2metrics(costs, iwa, expected_upper_bounds=expected_upper_bounds)

    s = ' & '.join(['{0:.3f}'.format(m) for m in metrics]) + ' \\\\'
    return s

##############################################################################
##############################################################################
##############################################################################

if __name__ == '__main__':
    # Running the experiments
    # Expected cwd: mhr/MFF-MUSCIMA-eval

    # Read annotation.
    annots_file = os.path.join(os.path.dirname(__file__),
                               '..', '..', '..',
                               'annotations', 'annotations.csv')
    annots = read_annotations(annots_file)

    # Use only annotators who did at least 98 cases.
    per_user = collect_per_annotator(annots)
    full_annots = reconstruct_from_per_annotator(per_user,
                                                 names=[n for n in per_user
                                                        if get_n_rankings(per_user[n]) >= 98],
                                                 exclude=False)

    # Filter out control group
    noctrl_annots = extract_control_group(full_annots, exclude=True)
    noctrl_iwa = item_wise_averages(noctrl_annots)
    noctrl_per_user = collect_per_annotator(noctrl_annots)

    # Random annotators:
    N_RANDOM_ANNOTATORS = 10
    wr_annots, rnames = update_with_random(noctrl_annots, N_RANDOM_ANNOTATORS)


    names = sorted(noctrl_per_user.keys())
    K = len(names)
    qs = numpy.arange(0.001, 0.5, 0.01)
    rC = annots2rC(noctrl_annots)

    pertrubations = numpy.arange(0.0, 0.51, 0.01)
    s_pertrubations = pertrubations[[0, 5, 10, 20, 30, 40, 50]]

    # This much we know:
    q_star = 0.13
    T_star_threshold = 0.34
    certain_examples = {e: i for e, i in flatten_annots_to_examples(noctrl_iwa).items() if i > T_star_threshold}


    # Compute full pairwise agreements
    print('Computing pairwise matrices...')
    wr_matrix = annots2pairwise_agreement_profile_matrix(wr_annots, weighed=True)
    ur_matrix = annots2pairwise_agreement_profile_matrix(wr_annots, weighed=False)

    w_matrix = wr_matrix[:K,:K]
    u_matrix = ur_matrix[:K,:K]

    # 1) Variance of the agreement metric \hat{L}_w(a, b)

    print('--------------------------------------------')
    print('Measuring agreement variances:')

    # - Prepare the groups to sample. Make sure the names are always sorted.
    print('Preparing annotator groups to sample.')
    N_GROUPS_SAMPLED = 100
    GROUP_SIZE = K / 2
    all_groups = map(tuple, map(sorted, list(itertools.combinations(names, K / 2))))
    random.shuffle(all_groups)
    sample_groups = all_groups[:N_GROUPS_SAMPLED]

    # - Compute the agreement matrix for the sampled groups
    w_matrix_per_group = {}
    u_matrix_per_group = {}

    print('Computing weighed and unweighed agreement matrices for annotator groups.')
    for i, g in enumerate(sample_groups):
        annots_g = filter_annotators(noctrl_annots, names=g, exclude=False)
        # annots_g_prime = filter_annotators(noctrl_annots, names=g, exclude=True)
        if (i % 10) == 0:
            print('At group {0} ({1})'.format(i, g))
        w_matrix_g = annots2pairwise_agreement_profile_matrix(annots_g, weighed=True)
        w_matrix_per_group[g] = w_matrix_g
        u_matrix_g = annots2pairwise_agreement_profile_matrix(annots_g, weighed=False)
        u_matrix_per_group[g] = u_matrix_g

    # - Collect the sample set per annotator pair.
    #   (Note: because the groups are chosen randomly, some pairs will
    #   have been seen more often than others.)
    #
    #   Naming convention:
    #       w_(...something...) stands for weighed agreement (low-consensus items are less important)
    #       u_(...something...) stands for unweighed agreement
    print('Aggregating agreement samples over annotator pairs.')
    w_samples_per_pair = {}  # All agreement scores of the given annotator pair, one from each group.
    u_samples_per_pair = {}
    for g, m_w, in w_matrix_per_group.items():
        m_u = u_matrix_per_group[g]
        for i, j in map(sorted, list(itertools.combinations(range(len(g)), 2))):
            a = g[i]
            b = g[j]
            if (a, b) not in w_samples_per_pair: w_samples_per_pair[(a, b)] = []
            w_ij = m_w[i][j]
            w_samples_per_pair[(a, b)].append(w_ij)
            if (a, b) not in u_samples_per_pair: u_samples_per_pair[(a, b)] = []
            u_ij = m_u[i][j]
            u_samples_per_pair[(a, b)].append(u_ij)

    for pair, w in w_samples_per_pair.items():
        w_samples_per_pair[pair] = numpy.array(w)
    for pair, u in u_samples_per_pair.items():
        u_samples_per_pair[pair] = numpy.array(u)

    w_var_per_pair = {p: numpy.var(w_samples_per_pair[p]) for p in w_samples_per_pair}
    w_mean_per_pair = {p: numpy.average(w_samples_per_pair[p]) for p in w_samples_per_pair}
    u_var_per_pair = {p: numpy.var(u_samples_per_pair[p]) for p in u_samples_per_pair}
    u_mean_per_pair = {p: numpy.average(u_samples_per_pair[p]) for p in u_samples_per_pair}

    w_mean_per_pair_matrix = numpy.ones((len(names), len(names)))
    u_mean_per_pair_matrix = numpy.ones((len(names), len(names)))
    w_var_per_pair_matrix = numpy.zeros((len(names), len(names)))
    u_var_per_pair_matrix = numpy.zeros((len(names), len(names)))
    for i, n in enumerate(names):
        for j, m in enumerate(names):  # Keeping the pairs (n, m) ordered.
            if m == n:
                continue
            w_mean_per_pair_matrix[i,j] = w_mean_per_pair[tuple(sorted((n, m)))]
            u_mean_per_pair_matrix[i,j] = u_mean_per_pair[tuple(sorted((n, m)))]
            w_var_per_pair_matrix[i,j] = w_var_per_pair[tuple(sorted((n, m)))]
            u_var_per_pair_matrix[i,j] = u_var_per_pair[tuple(sorted((n, m)))]


    print('Aggregating agreements by user.')
    # - Collect agreement stats per annotator, against everyone.
    #   Full sets -- no averaging yet. Flatten the agreements afterward.
    w_per_user = {}
    u_per_user = {}
    for a, b in w_samples_per_pair:
        if a not in w_per_user: w_per_user[a] = []
        if b not in w_per_user: w_per_user[b] = []
        w_per_user[b].append(w_samples_per_pair[(a, b)])
        w_per_user[a].append(w_samples_per_pair[(a, b)])
        if a not in u_per_user: u_per_user[a] = []
        if b not in u_per_user: u_per_user[b] = []
        u_per_user[b].append(u_samples_per_pair[(a, b)])
        u_per_user[a].append(u_samples_per_pair[(a, b)])
    for n, w in w_per_user.items():
        w_per_user[n] = numpy.array([x for arr in w_per_user[n] for x in arr])
        u_per_user[n] = numpy.array([x for arr in u_per_user[n] for x in arr])
    w_mean_per_user = {n: numpy.average(w_per_user[n]) for n in w_per_user}
    w_var_per_user = {n: numpy.var(w_per_user[n]) for n in w_per_user}
    u_mean_per_user = {n: numpy.average(u_per_user[n]) for n in u_per_user}
    u_var_per_user = {n: numpy.var(u_per_user[n]) for n in u_per_user}

    ##############################################################################
    ##############################################################################
    ##############################################################################

    # Preparing a cost assessment metric (Spearman's r, Pearson's r):
    #
    # 1. Upper bound: a metric composed of people
    # -------------------------------------------
    #
    #  - For each group g:
    print('Computing upper limits of cost function assessment metrics'
          ' (Spearman\'s r, Pearson\'s r, Kendall\'s Tau).'
          ' Computed from sampled annotator groups.')
    MAX_IWA_EPS = 0.00001
    spearman_per_group = {}
    spearman_per_true_per_group = {}
    pearson_per_group = {}
    pearson_per_true_per_group = {}
    kendall_per_group = {}
    kendall_per_true_per_group = {}
    rmse_per_group = {}
    rmse_per_true_per_group = {}
    for g in sample_groups:
        g_prime = [n for n in names if n not in g]
        # - Get IWAs from them and g_prime
        g_iwa = item_wise_averages(filter_annotators(noctrl_annots, names=g, exclude=False))
        g_prime_iwa = item_wise_averages(filter_annotators(noctrl_annots, names=g, exclude=True))
        # - match IWAs from the examples against each other (see: cds, iwas)
        g_i, g_prime_i = cost_diffs_vs_iwas(g_iwa, g_prime_iwa)
        # - Introducing some minimal noise, to avoid zero-variance errors... messes up the ranks.
        #   Instead, if variance of one is 0, find argmax of other and add a little.
        if g_i.max() - g_i.min() == 0:
            g_prime_argmax = numpy.argmax(g_prime_i)
            g_i[g_prime_argmax] += 0.00001
        if g_prime_i.max() - g_prime_i.min() == 0:
            g_argmax = numpy.argmax(g_i)
            g_prime_i[g_argmax] += 0.00001
        #g_i *= numpy.random.uniform(1 - MAX_IWA_EPS, 1.0, size=g_i.shape)
        #g_prime_i *= numpy.random.uniform(1 - MAX_IWA_EPS, 1.0, size=g_prime_i.shape)
        # - compute assessment metrics
        spearman_per_group[g] = scipy.stats.spearmanr(g_i, g_prime_i).correlation
        pearson_per_group[g] = scipy.stats.pearsonr(g_i, g_prime_i)[0]
        kendall_per_group[g] = scipy.stats.kendalltau(g_i, g_prime_i).correlation
        rmse_per_group[g] = numpy.sqrt(((g_prime_i - g_i) ** 2).mean())
        # Compute pearson & spearman separately for each true-based group, get weighed average?
        for a in g_iwa:
            if a not in spearman_per_true_per_group: spearman_per_true_per_group[a] = {}
            if a not in pearson_per_true_per_group: pearson_per_true_per_group[a] = {}
            if a not in kendall_per_true_per_group: kendall_per_true_per_group[a] = {}
            if a not in rmse_per_true_per_group: rmse_per_true_per_group[a] = {}
            g_a_iwa = filter_annots_by_true(g_iwa, [a], exclude=False)
            g_prime_a_iwa = filter_annots_by_true(g_prime_iwa, [a], exclude=False)
            g_a_i, g_prime_a_i = cost_diffs_vs_iwas(g_a_iwa, g_prime_a_iwa)
            if numpy.max(g_a_i) - numpy.min(g_a_i) == 0:
                g_prime_a_argmax = numpy.argmax(g_prime_a_i)
                g_a_i[g_prime_a_argmax] += 0.00001
            if numpy.max(g_prime_i) - numpy.min(g_prime_i) == 0:
                g_a_argmax = numpy.argmax(g_a_i)
                g_prime_a_i[g_a_argmax] += 0.00001
            spearman_per_true_per_group[a][g] = scipy.stats.spearmanr(g_a_i, g_prime_a_i).correlation
            pearson_per_true_per_group[a][g] = scipy.stats.pearsonr(g_a_i, g_prime_a_i)[0]
            kendall_per_true_per_group[a][g] = scipy.stats.kendalltau(g_a_i, g_prime_a_i).correlation
            rmse_per_true_per_group[a][g] = numpy.sqrt(((g_prime_a_i - g_a_i) ** 2).mean())

    # Overall average s, p, k. Filter out NaN.
    s_values = numpy.array(spearman_per_group.values())
    spearman_average = numpy.average(s_values[numpy.isfinite(s_values)])
    spearman_stddev = numpy.std(s_values[numpy.isfinite(s_values)])
    p_values = numpy.array(pearson_per_group.values())
    pearson_average = numpy.average(p_values[numpy.isfinite(p_values)])
    pearson_stddev = numpy.std(p_values[numpy.isfinite(p_values)])
    k_values = numpy.array(kendall_per_group.values())
    kendall_average = numpy.average(k_values[numpy.isfinite(k_values)])
    kendall_stddev = numpy.std(k_values[numpy.isfinite(k_values)])
    rmse_values = numpy.array(rmse_per_group.values())
    rmse_average = numpy.average(rmse_values[numpy.isfinite(rmse_values)])
    rmse_stddev = numpy.std(rmse_values[numpy.isfinite(rmse_values)])


    print('Aggregating agreement metric upper limits by True score.')
    # Factorized into groups by true score:
    spearman_per_true_average = {}
    spearman_per_true_stddev = {}
    pearson_per_true_average = {}
    pearson_per_true_stddev = {}
    kendall_per_true_average = {}
    kendall_per_true_stddev = {}
    rmse_per_true_average = {}
    rmse_per_true_stddev = {}
    for a in spearman_per_true_per_group:
        s = numpy.array(spearman_per_true_per_group[a].values())
        spearman_per_true_average[a] = numpy.average(s[numpy.isfinite(s)])
        spearman_per_true_stddev[a] = numpy.std(s[numpy.isfinite(s)])
        p = numpy.array(pearson_per_true_per_group[a].values())
        pearson_per_true_average[a] = numpy.average(p[numpy.isfinite(p)])
        pearson_per_true_stddev[a] = numpy.std(p[numpy.isfinite(p)])
        k = numpy.array(kendall_per_true_per_group[a].values())
        kendall_per_true_average[a] = numpy.average(k[numpy.isfinite(k)])
        kendall_per_true_stddev[a] = numpy.std(k[numpy.isfinite(k)])
        rmse = numpy.array(rmse_per_true_per_group[a].values())
        rmse_per_true_average[a] = numpy.average(rmse[numpy.isfinite(rmse)])
        rmse_per_true_stddev[a] = numpy.std(rmse[numpy.isfinite(rmse)])

    # 2. Lower bound:
    # ---------------
    #
    # - Random metric against group consensus
    # - Constant metric against group consensus
    # - Constant metric with noise against group consensus


    # Processing a cost:
    #  - load the costs
    #  - filter them to the non-control cases
    #  - Per group: compute Spearman's rank, Pearson's rank


    ##############################################################################
    ##############################################################################
    ##############################################################################

    # # Binomial-based math.
    # print('with_per')
    # all_ps_with_per = [p_rC(rC, noctrl_iwa, names=names, qs=qs, simplified=True, TC_0_pertrubation=p_per)
    #                    for p_per in s_pertrubations]
    # print('with_nper')
    # all_ps_with_nper = [p_rC(rC, noctrl_iwa, names=names, qs=qs, simplified=True, TC_non0_pertrubation=p_per)
    #                     for p_per in s_pertrubations]
    # print('with_both')
    # all_ps_with_bothper = [p_rC(rC, noctrl_iwa, names=names, qs=qs, simplified=True, TC_0_pertrubation=p_per / 2.0, TC_non0_pertrubation=p_per / 2.0)
    #                        for p_per in s_pertrubations]
    # print('random (stays out of combination)')
    # all_ps_with_random = [p_rC(rC, noctrl_iwa, names=names, qs=qs, simplified=True, n_randomize_TCs=100)
    #                       for _ in xrange(10)]
    #
    # all_ps = numpy.array(all_ps_with_per + all_ps_with_nper + all_ps_with_bothper)
    # all_ps_sum = numpy.sum(all_ps, axis=0)
    # print('done')
    #
    # s_log10_per_modes = [numpy.log10(p.max()) for p in all_ps_with_per]
    # s_log10_nper_modes = [numpy.log10(p.max()) for p in all_ps_with_nper]
    # s_log10_bothper_modes = [numpy.log10(p.max()) for p in all_ps_with_bothper]
    # s_log10_random_modes = [numpy.log10(p.max()) for p in all_ps_with_random]
    # s_log10_all_nodes = [numpy.log10(p.max()) for p in all_ps]
    #
    # # Relationship between thresholds and q*
    # noctrl_iwa_abs_values = sorted(set(numpy.absolute(numpy.array(collect_leaf_values(noctrl_iwa)))))
    # q_opt_per_iwa = collections.OrderedDict()
    # s_log10_per_iwa = collections.OrderedDict()
    # for thr_idx, iwa in enumerate(noctrl_iwa_abs_values):
    #     q_argmax = all_ps_sum[:,thr_idx].argmax()
    #     q_opt_per_iwa[iwa] = qs[q_argmax]
    #     s_log10_per_iwa[iwa] = numpy.log10(all_ps_sum[q_argmax, thr_idx])
    #
    #


    # Now what?
    #
    # - Assign T^*
    # - Compute agreement
    # - Estimate agreement
    # - Remove petr.haas and re-compute agreement
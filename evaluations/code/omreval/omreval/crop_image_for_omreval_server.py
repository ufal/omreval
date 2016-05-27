#!/usr/bin/env python
"""This is a script that crops a PNG image to the given margin. All
non-white areas are considered relevant and not to be cropped out.
"""
from __future__ import print_function
import argparse
import logging
import time

import numpy
import cv2
import matplotlib.pyplot as plt

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store',
                        help='The input PNG file.')
    parser.add_argument('-o', '--output', action='store',
                        help='The output PNG file.')
    parser.add_argument('--margin', type=int, default=10, action='store',
                        help='Width of margin around black area.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Your code goes here
    color_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    image = numpy.zeros(color_image.shape[:2], dtype='uint8')
    image[color_image[:, :, 3] == 0] = 255
    logging.info('Color shape: {0}, gray shape: {1}'.format(color_image.shape, image.shape))

    logging.info('Image shape: {0}'.format(image.shape))
    logging.info('Image max: {0}'.format(image.max()))
    if args.debug:
        plt.imshow(color_image)
        plt.imshow(image, cmap='gray')
        plt.show()
    mask = numpy.zeros(image.shape, dtype='uint8')
    mask[image != image.max()] = 1  # Darker areas
    colsum = mask.sum(axis=0)
    rowsum = mask.sum(axis=1)

    left = 0
    right = image.shape[1]
    _found_left = False
    for i in xrange(image.shape[1]):
        if colsum[i] == 0:
            if not _found_left:
                continue
            elif colsum[i:].sum() == 0:
                right = i
                break
        else:
            if not _found_left:
                left = i
                _found_left = True
            else:
                continue

    top = 0
    bottom = image.shape[0]
    _found_top = False
    for i in xrange(image.shape[0]):
        if rowsum[i] == 0:
            if not _found_top:
                continue
            elif rowsum[i:].sum() == 0:
                bottom = i
                break
        else:
            if not _found_top:
                top = i
                _found_top = True
            else:
                continue

    top = max(top - args.margin, 0)
    left = max(left - args.margin, 0)
    bottom = min(bottom + args.margin, image.shape[0])
    right = min(right + args.margin, image.shape[1])
    logging.info('Bounding box: {0}:{1}, {2}:{3}'.format(top, bottom, left, right))

    output = color_image[top:bottom, left:right, :]
    if args.debug:
        plt.imshow(output)
        plt.show()
    cv2.imwrite(args.output, output)

    _end_time = time.clock()
    logging.info('crop_image_for_omreval_server.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

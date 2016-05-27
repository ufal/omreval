"""This module implements tools for loading the CVC-MUSCIMA and MUSCIMA++
datasets."""
from __future__ import print_function, unicode_literals
import logging
import os
import cv2
import numpy
from lxml import etree

logger = logging.getLogger(__name__)

__author__ = "Jan Hajic jr."

if 'CVC_MUSCIMA_ROOT' in os.environ:
    CVC_MUSCIMA_ROOT = os.environ['CVC_MUSCIMA_ROOT']
else:
    CVC_MUSCIMA_ROOT = os.path.join('/Users', 'hajicj', 'data', 'CVC-MUSCIMA')

CVC_MUSCIMA_PRINTED = os.path.join(CVC_MUSCIMA_ROOT, 'printed')
CVC_MUSCIMA_HANDWRITTEN = os.path.join(CVC_MUSCIMA_ROOT, 'handwritten')

CVC_MUSCIMA_DISTORTIONS = [
    'curvature',
    'ideal',
    'interrupted',
    'kanungo',
    'rotated',
    'staffline-thickness-variation-v1',
    'staffline-thickness-variation-v2',
    'staffline-y-variation-v1',
    'staffline-y-variation-v2',
    'thickness-ratio',
    'typeset-emulation',
    'whitespeckles',
]

CVC_MUSCIMA_STAFFONLY_SUBDIR = 'gt'
CVC_MUSCIMA_SYMBOLS_SUBDIR = 'symbol'
CVC_MUSCIMA_FULL_SUBDIR = 'image'

if not 'MFF_MUSCIMA_ROOT' in os.environ:
    MFF_MUSCIMA_ROOT = os.path.join('/Users', 'hajicj', 'mhr', 'MFF-MUSCIMA')
else:
    MFF_MUSCIMA_ROOT = os.environ['MFF_MUSCIMA_ROOT']


######################################################
# Utility functions for name/writer conversions
_hex_tr = {
    '0': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
}
def parse_hex(hstr):
    """Convert a hexadecimal number string to integer.

    >>> parse_hex('33')
    51
    >>> parse_hex('abe8')
    44008

    """
    out = 0
    for i, l in enumerate(reversed(hstr)):
        out += (16**i) * _hex_tr[l]
    return out


def hex2rgb(hstr):
    """Parse a hex-coded color like '#AA0202' into a floating-point representation."""
    if hstr.startswith('#'):
        hstr = hstr[1:]
    rs, gs, bs = hstr[:2], hstr[2:4], hstr[4:]
    r, g, b = parse_hex(rs), parse_hex(gs), parse_hex(bs)
    return r / 255.0, g / 255.0, b / 255.0


def cvc_musicma_number2printed(n, suffix='png'):
    if (n < 1) or (n > 20):
        raise ValueError('Invalid MUSCIMA score number {0}.'
                         ' Valid only between 1 and 20.'.format(n))
    if n < 10:
        return 'F' + '0' + str(n) + '.' + suffix
    else:
        return 'F' + str(n) + '.' + suffix


def cvc_muscima_number2handwritten(n):
    if (n < 1) or (n > 20):
        raise ValueError('Invalid MUSCIMA score number {0}.'
                         ' Valid only between 1 and 20.'.format(n))
    if n < 10:
        return 'p0' + '0' + str(n) + '.png'
    else:
        return 'p0' + str(n) + '.png'


def cvc_muscima_number2writer(n):
    if (n < 1) or (n > 50):
        raise ValueError('Invalid MUSCIMA writer number {0}.'
                         ' Valid only between 1 and 50.'.format(n))
    if n < 10:
        return 'w-0' + str(n)
    else:
        return 'w-' + str(n)


########################################################
# Loading MUSCIMA images

class MUSCImage(object):
    """The class to access one MUSCIMA score by a writer in its various forms:

    * Full
    * No staves
    * Staves only
    * Printed form (full only)

    >>> img = MUSCImage(number=10, writer=32, distortion='curvature')
    >>> p = img.printed
    >>> f = img.full
    >>> gt = img.staff_only
    >>> s = img.symbols

    Over various writers and the printed version, the MUSCImages are
    aggregated by MUScore.
    """

    def __init__(self, number, writer, root=CVC_MUSCIMA_ROOT,
                 distortion='ideal'):
        """Initialize the image paths.

        :param number: between 1 and 20 (inclusive).

        :param writer: between 1 and 50 (inclusive).

        :param root: The path to the CVC-MUSCIMA root.

        :param distortion: The type of distortion applied. By default,
            use the non-distorted images.
        """
        if distortion not in CVC_MUSCIMA_DISTORTIONS:
            raise ValueError('Invalid distortion: {0}'.format(distortion))

        # Printed score: determine whether it has a *.png or *.tiff suffix.
        printed_fname = cvc_musicma_number2printed(number, suffix='png')
        if not os.path.isfile(os.path.join(CVC_MUSCIMA_PRINTED, printed_fname)):
            printed_fname = cvc_musicma_number2printed(number, suffix='tiff')

        self.printed_path = os.path.join(CVC_MUSCIMA_PRINTED, printed_fname)
        self._printed = None

        # Handwritten root
        writer_dirname = cvc_muscima_number2writer(writer)
        hw_fname = cvc_muscima_number2handwritten(number)
        hw_root = os.path.join(CVC_MUSCIMA_HANDWRITTEN, distortion, writer_dirname)

        self._staff_only = None
        self.staff_only_path = os.path.join(hw_root, CVC_MUSCIMA_STAFFONLY_SUBDIR,
                                            hw_fname)

        self._symbols = None
        self.symbols_path = os.path.join(hw_root, CVC_MUSCIMA_SYMBOLS_SUBDIR,
                                         hw_fname)

        self._full = None
        self.full_path = os.path.join(hw_root, CVC_MUSCIMA_FULL_SUBDIR,
                                      hw_fname)

    @property
    def printed(self):
        return self._image_getter('printed')

    @property
    def full(self):
        return self._image_getter('full')

    @property
    def staff_only(self):
        return self._image_getter('staff_only')

    @property
    def symbols(self):
        return self._image_getter('symbols')

    def _image_getter(self, varname):
        container_name = '_' + varname
        if not hasattr(self, container_name):
            raise AttributeError('Cannot get image {0}, container variable {1}'
                                 ' not defined.'.format(varname, container_name))
        container = getattr(self, container_name)
        if container is not None:
            return container

        path_name = varname + '_path'
        if not hasattr(self, path_name):
            raise AttributeError('Cannot load image {0}, container variable is'
                                 ' not initialized and path variable {1}'
                                 ' is not defined.'.format(varname, path_name))
        path = getattr(self, path_name)
        if path is None:
            raise ValueError('Cannot load image {0} from path, path variable'
                             ' {1} is not initialized.'.format(varname, path_name))
        img = self._load_grayscale_image(path)
        setattr(self, container_name, img)

        return img

    @staticmethod
    def _load_grayscale_image(path):
        img = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
        return img

    def symbol_bboxes(self, with_labels=False):
        """Extracts bounding boxes from symbols image."""
        cc, labels = cv2.connectedComponents(self.symbols)
        bboxes = {}
        for x, row in enumerate(labels):
            for y, l in enumerate(row):
                if l not in bboxes:
                    bboxes[l] = [x, y, x+1, y+1]
                else:
                    box = bboxes[l]
                    if x < box[0]:
                        box[0] = x
                    elif x + 1 > box[2]:
                        box[2] = x + 1
                    if y < box[1]:
                        box[1] = y
                    elif y + 1 > box[3]:
                        box[3] = y + 1

        if with_labels:
            return bboxes, labels
        else:
            return bboxes

    def symbol_crops(self):
        """Extract the cropped symbols from the symbols image."""
        bboxes = self.symbol_bboxes()
        s = self.symbols
        crops = []
        for t, l, b, r in bboxes.values():
            crops.append(s[t:b,l:r])
        return crops


##############################################################################

# Goal: render annotations on the page (in their original color).

# Annotations from MUSCIMA++
class CropObject(object):
    """One annotated object."""
    def __init__(self, objid, clsid, x, y, width, height):
        self.objid = objid
        self.clsid = clsid
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.top = self.x
        self.bottom = self.x + height
        self.left = self.y
        self.right = self.y + width

    def render(self, img, alpha=0.3, rgb=(1.0, 0.0, 0.0)):
        """Renders itself upon the given image as a rectangle
        of the given color and transparency.

        :param img: A three-channel image (3-D numpy array,
            with the last dimension being 3)."""
        color = numpy.array(rgb)
        mask = numpy.ones((self.height, self.width, 3)) * color
        crop = img[self.top:self.bottom, self.left:self.right]
        mix = (crop + alpha * mask) / (1 + alpha)

        img[self.top:self.bottom, self.left:self.right] = mix
        return img


class MLClass(object):
    """Information about the annotation class. We're using it
    mostly to get the color of rendered CropObjects."""
    def __init__(self, clsid, name, folder, color):
        self.clsid = clsid
        self.name = name
        self.folder = folder
        # Parse the string into a RGB spec.
        r, g, b = hex2rgb(color)
        self.color = (r, g, b)


def parse_cropobject_list(filename):
    """From a xml file with a CropObjectList as the top element,
    """
    tree = etree.parse(filename)
    root = tree.getroot()
    cropobject_list = []
    for cropobject in root.iter('CropObject'):
        obj = CropObject(objid=int(cropobject.findall('Id')[0].text),
                         clsid=int(cropobject.findall('MLClassId')[0].text),
                         x=int(cropobject.findall('Y')[0].text),
                         y=int(cropobject.findall('X')[0].text),
                         width=int(cropobject.findall('Width')[0].text),
                         height=int(cropobject.findall('Height')[0].text))
        cropobject_list.append(obj)

    return cropobject_list


def parse_mlclass_list(filename):
    """From a xml file with a MLClassList as the top element,
    """
    tree = etree.parse(filename)
    root = tree.getroot()
    mlclass_list = []
    for mlclass in root.iter('MLClass'):
        obj = MLClass(clsid=int(mlclass.findall('Id')[0].text),
                      name=mlclass.findall('Name')[0].text,
                      folder=mlclass.findall('Folder')[0].text,
                      color=mlclass.findall('Color')[0].text)
        mlclass_list.append(obj)
    return mlclass_list


def render_annotations(img, cropoboject_list, mlclass_list, alpha=1.0):
    """Render the annotation bounding boxes of the given cropobjects
    onto the img. Take care to load the same image that was annotated,
    and to load the correct MLClassList to get names & colors!

    :param img: The image is expected to be in 3-channel RGB mode, floating-point,
        within ``(0.0, 0.1)``.

    :param alpha: Render with this weight of the bounding box colored rectangles.
        Set alpha=1.0 for a 50:50 img/boundingbox color mix. Note that at the end,
        the rendering is averaged with the original image again, to accent
        the actual notation.
    """
    mlclass_dict = {m.clsid: m for m in mlclass_list}

    output = img * 1.0
    for obj in cropoboject_list:
        rgb = mlclass_dict[obj.clsid].color
        obj.render(output, alpha=alpha, rgb=rgb)

    return (output + img) / 2.0

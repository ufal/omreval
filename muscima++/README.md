
This directory contains the MUSCIMA++ dataset for benchmarking
Optical Music Recognition.

We do not store the underlying CVC-MUSCIMA data here, to avoid
duplication. Download them from the official CVC-MUSCIMA page.

Directories and files
---------------------

...



Dataset
==========================

How the OMR dataset we are building is provided:

**MUSCIMA++** is the dataset we are in the process of building.
It is divided into two general levels:

* **CVC-MUSCIMA**, which is the original database of 20 scores transcribed
  by 50 musicians, done by Alicia Fornés et al. from the Computer Vision Center (CVC)
  at Universitat Autònoma de Barcelona. The scores then went through 12 different
  distortions, as defined by Dalitz et al. (2008).
* **MFF-MUSCIMA**, which is the gradual enrichment of CVC-MUSCIMA by symbol-level
  and MusicXML annotation. (This enrichment process is why we call the total
  *MUSCIMA++* -- consulting with Dr. Fornés, we decided that CVC-MFF-MUSCIMA
  doesn't quite roll off the tongue...)

This dataset was intended to serve for competitions in staff removal
and writer identification. Therefore, in the OMR pipeline, it has ground
truth for *staff removal* and is not annotated at higher levels (symbols
and end-to-end transcription in MusicXML). However, the underlying data
provides a perfect mix of various CWMN symbols and structures, a good
range of different writing styles, and at the same thousands of symbols
from each individual writer, so we believe it will make for a very robust
OMR dataset.

We do not duplicate the CVC-MUSCIMA dataset in this repository:
for one, the total of 12000 images is quite large for Git. Therefore,
you will only find the **MFF-MUSCIMA** portion, and its organization into
its own layers of ground truth (symbolic and MusicXML). To construct a complete
MUSCIMA++ dataset on your machine, just make your directory structure
look like this:

    MUSCIMA++/
         |
         +-- CVC-MUSCIMA
         +-- MFF-MUSCIMA

The dataset is organized into *layers*, corresponding to top-level directories
in the `data/` subfolder. There is currently a *Symbolic* layer and an *XML*
layer. The XML layer consists of the MusicXML transcription of scores `F01`, `F03`,
`F08`, `F10` and `F16` from the CVC-MUSCIMA dataset. The symbolic layer consists
of these scores annotated at the *notation primitive* level.

The layers are described in detail by their own `README.md` files;
refer to these for details.




Code
==========================

The Python interface to the dataset is organized into a Python package `muscima`
that can be installed like any other package through `setup.py`.

The `muscima` module inside the package implements an interface for a score
image in CVC-MUSCIMA in the `MUSCImage` class. The functionality is mostly
rudimentary, but we are working on expanding it. It encapsulates access
to one image (indexed by one score, one writer, one type of distortion)
in its various versions (full, symbols only, staff only, printed original).
It similarly provides Python abstractions for the annotations at the XML
and symbolic levels, together with parsing capabilities. Refer to the tutorial
`muscima++.ipynb` for examples of usage.



Requirements
---------------------------

In order to use the `muscima` package, you will need Python 2.7.11 and
newer. We strongly recommend using the Anaconda distribution, which is
freely available for academic use, and using a `conda`-based virtual
environment to replicate our experiments, so as to prevent interference
with your own Python environment(s).

Python packages:

* Scientific Python packages minimum: numpy, scipy, matplotlib
  (Anaconda should take care of these for you, in case you don't have them.)
* OpenCV3 (Implements connected component search and image loading
  and saving, plus it's a good idea to have OpenCV installed anyway.)

Then, download the CVC-MUSCIMA dataset and set the `CVC_MUSCIMA_ROOT`
to point to its top-level directory.


Installation
---------------------------

Go to `code/muscima` and run `python setup.py install`. If you did not
set up a virtual environment for working with our code, consider running
`python setup.py develop` which only symlinks the modules and scripts
instead of copying them over to your Python library & scripts directory,
thus not polluting your installation as much.

If you have `pip` (which you should, and if you are using
Anaconda, you do have it), use `pip wheel .` and `pip install`.



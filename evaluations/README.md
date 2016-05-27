


Licence information
==========================

Please see the LICENSE file in the root directory of the supplementary
materials.

The data is released under a Creative Commons-Attribution licence.

The code is released under the MIT licence, with some exceptions
where we tie in GPL technologies. The exceptions are detailed
in the LICENSE file in the root directory of the supplementary materials.

"Data" is everything in the `annotations/`, `costs/` and `test_case_corpus/`
subdirectories.

"Code" is everything in the `code/` subdirectory.

Again, please see LICENSE for details. The licensing situation is a little
complicated because we tie together many technologies with different
licensing requirements.






Directories and files
==========================

* **annotations/** -- contains the anonymized raw annotation data
* **code/** -- the code to run the various experiments.
* **costs/** -- contains the costs of individual system outputs measured by the
  various evaluation metrics described in the Article.
* **test_case_corpus/** -- contains the MusicXML representations of the test cases
  ( **MusicXML/**) and the rendered images that were shown to annotators ( **images**).






Code
==========================

The evaluation code is organized into a Python package `omreval`
that can be installed like any other package through `setup.py`.
The scripts that implement the individual cost functions are installed
as scripts.

The `omreval` package contains a module with annotation processing functions,
`process_annotations.py`, and an implementation of the automated OMR
evaluation metrics that were proposed in the article. There are two helper
modules for the evaluation metrics: `pitch_counter.py`, which implements
hashing MusicXML `<pitch>` elements as single characters, and `zss_metrics.py`,
which implements edit costs for the tree edit distance-based metrics
(TED and TEDn in the article). Finally, scripts are available for
exporting the MusicXML test cases in the corpus as images, using
`export_images_for_omreval_server.sh` and the helper script
`crop_image_for_omreval_server.py`.

All Python and shell scripts will print their documentation and usage
instruction when run with `-h`.
Refer to the tutorial `omreval_tutorial.ipynb` for examples of usage
of the `process_annotations` module.



Requirements
---------------------------

In order to use the `omreval` package, you will need Python 2.7.11 and
newer. We strongly recommend using the Anaconda distribution, which is
freely available for academic use, and using a `conda`-based virtual
environment to replicate our experiments, so as to prevent interference
with your own Python environment(s).

Python packages:

* Scientific Python packages minimum: numpy, scipy, matplotlib
  (Anaconda should take care of these for you, in case you don't have them.)
* Levenshtein
* zss (https://github.com/timtadh/zhang-shasha)
  Implements tree edit distance using the Zhang-Shasha algorithm
* OpenCV3 (implements connected component search when postprocessing
  the images rendered from the test case corpus)

Other software:

* MuseScore (for converting MusicXML into PNG images)
* LilyPond (for the **Ly** evaluation metric, and great music typesetting!)



Installation
---------------------------

Go to `code/omreval` and run `python setup.py install`. If you did not
set up a virtual environment for working with our code, consider running
`python setup.py develop` which only symlinks the modules and scripts
instead of copying them over to your Python library & scripts directory,
thus not polluting your installation as much.

If you have `pip` (which you should, and if you are using
Anaconda, you do have it), use `pip wheel .` and `pip install`.




Test case corpus
==========================

**NOTE** Anytime some scripts ask you for the test case corpus root,
provide the path to `test_case_corpus/MusicXML`.

The test case corpus is organized into groups of scores according to which
"true" score they are derived from. The true scores are distinguished by
a `_true` suffix in their filename.
There is generally one true score per directory, with the exception
of the **complex/** subdir, which has three, and **multi-part/**, which
has two.
The filenames try to be descriptive: they reflect the kind of error
we wanted present and ranked relative to others.



Filename syntax
--------------------------

`score-group-name_type-of-error.{xml,png}`

For each `score-group-name`, there will be a `score-group-name_true.{xml,png}`
file with the true score for that group.



Corpus composition
--------------------------

The test case preference judgments are meant to constrain the space
of OMR evaluation metrics to those that behave sensibly with
respect to human cost-to-correct. We use human estimation of relative
preference as a proxy for measuring the *true* cost-to-correct.

The test cases were selected to cover a wide range of errors
that still result in a syntactically correct score. In addition
to various ways of types of error, it aims to show these errors in various
*contexts* in which they happen: a single note, a sequence of notes,
a two-part score (which includes segmentation errors), and longer
fragments, both single- and multi-voice and single- and multi-staff.

There are 8 "true" scores and 34 mangled "system outputs".
The best way to see what is actually in the corpus is to browse
the images that were shown to annotators (`test_case_corpus/images/`).






Costs
=========================

The `costs` directory stores the costs that our proposed evaluation metrics
assigned to the individual scores, and the **costs/cost-pairs.csv** file,
which is provided for convenience to replicate the results using our
implementations of the metrics.

In the process of assessing an automated OMR evaluation metric (=a cost
function that takes two MusicXML documents as inputs and returns some cost
of changing the second one to the first one), the cost files are
an intermediate-stage artifact: they collect the costs assigned by the metric
to the "system output" scores from our test case corpus.
To complete the cost function assessment, the costs of the
individual mangled scores are then compared against each other and these
differences are then correlated against the human annotators' preferences.

To match the costs to the metric assessment table in the article:

Filename | Metric
---------|--------
`costs_pure-Levenshtein.csv` | c14n
`costs_lilypond` | Ly
`costs_treedist-zss-unfiltered` | TED, without filtering auxiliary and MIDI nodes (not in article)
`costs_treedist-zss.csv` | TED
`costs_treedist-zss-Levenshtein.csv` | TEDn

The costs files are parsed by the `annot_processing.read_costs()` function
when evaluating how well an evaluation metric agrees with the human judgments.

The procedure for assessing a cost function against the human judgment data
is shown in a tutorial Jupyter notebook.



Implemented cost functions
---------------------------

We provide the implementations of the proposed cost functions (automated
OMR evaluation metrics) from the article as scripts in the `omreval`
package.

All scripts evaluate a single "true vs. prediction" item, using `-t` for
the true score and `-p` for the system output. The cost is printed to `stdout`.

Script (+parameters)                | Metric
------------------------------------|-------------
levenshtein_eval.py                 | c14n
lilypond_eval.py                    | Ly
treedist_eval.py -m zss             | TED
treedist_eval.py -m zss_Levenshtein | TEDn

You can use the `costs/cost-pairs.csv` file to iterate over the `-t`
and `-p` parameters.

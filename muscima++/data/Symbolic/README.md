MFF-MUSCIMA Symbol layer
========================

The Symbol layer is organized this way:

* ``data/``: Contains the XML classes data itself.
* ``specification/``:
    * ``mff-muscima-mlclasses.xsd``: The schema for the XML that defines
      symbol classes for OMR Toolbox.
    * ``mff-muscima-mlclasses.xml``: The XML that OMR Toolbox loads to know which
      symbol classes are defined for MFF-MUSCIMA.
    * ``mff-muscima-symbolilc.xsd``: The schema for the XML with the symbol data,
      exported by OMR Toolbox.
    * ``mff-muscima-mlclasses.csv``: A more maintainable way of keeping the list
      of symbol classes than the ``mff-muscima-mlclasses.xml`` document. Use the
      ``tools/classes2mlclasslist.py`` script to generate the XML from this file.
* ``tools/``:
    * ``classes2mlclasslist.py``: A script to convert the symbol class definition CSV
      into a OMR Toolbox-consumeable XML.



Annotation files
--------------------

There are three components to a set of annotations of a musical score:

* The symbol classes definition: **MLClassList**
* The image that is being annotated: **Image**
* The bounding boxes and their class labels: **CropObjectList**

The `MLClassList` and `CropObjecList` are represented as XML files,
specified by their respective XML schemas in the `specification/` subdirectory.
(For the sake of completeness: images are bitmaps.)

Each CropObjectList file contains a reference to the `MLClassList` and image
file to which it applies as a comment. The file paths in the comment are
relative to the root of MUSCIMA++. The comment is on the second line, with
the format:

`<!--Refs MLClassList="Symbolic/specification/some-mlclasses.xml" image="Symbolic/data/xyz.png"-->`

(In a future release, we will change this, most probably to a full-fledged XML element.)



Symbol definition
-----------------------

The CropObject is one musical symbol belonging to a class defined in the given
`MLClassList`. It is represented as a bounding box, using the lower left corner
and its width and height. The `CropObject` XML element is defined
in `specification/mff-muscima-symbolic.xsd`.

* `Id` is the ID of the given CropObject, unique to the file.
* `MLClassId` is a reference to the class of the given CropObject from the MLClassList file
  used for the annotation.
* `X` is the **horizontal** position of the lower left corner of the CropObject,
  measured from the **left**.
* `Y` is the **vertical** position of the lower left corner of the CropObject,
  measured from the **top**.
* `Width` is the width of the CropObject (how many columns does the bounding box have).
* `Height` is the height of the CropObject (how many rows does the bounding box have).

There is also an auxiliary `Selected` member element (we used it during annotation,
don't worry about it).



Class definition
-----------------------

The `MLClassList` XML is defined in `specification/mff-muscima-mlclasses.xsd`.
It defines the available symbol classes in a list of `MLClass` elements:

* `Id` is the identifier of the class for `CropObject`s labeled as this class.
  The text of `MLClassId` elements in a CropObjectList has to be an `Id` of
  a `MLClass` in the MLClassList file linked from the CropObjectList.
* `Name` is the human-readable label of the class, such as "notehead-full".

`Folder` and `Color` are export/rendering instructions for our annotation
tools; you can ignore them (although the colors are chosen, dare we say,
quite well).

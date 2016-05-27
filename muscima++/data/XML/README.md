MFF-MUSCIMA XML layer
=====================

The XML layer is organized this way:

* ``data/``: Contains the XML data itself. The subdirectories reflect stages of
  processing.
    * ``raw-mscz-export/``: MuseScore MusicXML export, without any postprocessing.
    * ``edited-mscz-export/``: The MusicXML with some manual editing, as some shortcomings
      of MuseScore export cannot be corrected automatically.
    * ``edited-postprocessed-mscz-export/``: MusicXML after automatic postprocessing.
    * ``mff-muscima-xml/``: After converting the MusicXML data into MFF-MUSCIMA format.
      This is the final stage.
* ``specification/``: Definitions for various xml formats. Currently
  irrelevant, but will contain the XSD for MFF-MUSCIMA in the future.
  MFF-MUSCIMA xml format is currently defined through the conversion
  scripts in tools as whatever MusicXML you pass to ``musicxml2mff.py``.
* ``tools/``: MFF-MUSCIMA-specific tools for handling MusicXML and
  conversion to/from our MFF-MUSCIMA-XML format.



Transcribing MusicXML for MUSCIMA++
------------------------------------

To get the XML data, we used the following process:

1. Transcribe into MuseScore.
2. Export MusicXML from MuseScore into ``data/raw-mscz-export``.
3. Apply manual corrections. Outputs go to ``data/edited-mscz-export``.
    1. Add part symbol information for multi-staff parts.
       Ex.: ``<part-symbol top-staff="1" bottom-staff="2">brace</part-group>``
    2. Check for repeat-bar signs and add appropriate ``<measure-style>``.
       (This symbol is not found in CVC-MUSCIMA, so we didn't have to do it,
       but if you encounter it, it needs to be done manually -- MuseScore
       only exports a ``<forward>`` element filling the whole bar.)
4. Apply automatic corrections using the ``tools/postprocess_mscz_export.py``
   script. Outputs go to ``data/edited-postprocessed-mscz-export``.
5. Convert to MFF-MUSCIMA-XML (adds ``eid="pref_[etype]_[enumber]_[gnumber]"``
   attributes to each element). Outputs go to `data/mff-muscima-xml`.

(All scripts have a `-h` option for help.)

After step 4, the file is still valid MusicXML: the postprocessing only fixes
some glitches in MuseScore export and fills in default elements (such as barlines)
that we will need for each symbol to have its MusicXML element counterpart.
After step 5, however, the `eid` attributes does break compatibility.

For transcription and export, we used MuseScore version 2.0.2, rev. f51dc11
(the stable release of MuseScore 2.0.2).
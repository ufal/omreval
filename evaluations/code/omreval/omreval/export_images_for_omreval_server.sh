#!/usr/bin/env bash
#
#

usage=$(cat <<"USAGE"
This script serves to export the test-data images (incl. directory structure)
as a set of PNG images for the evaluation server.

Supply the root of the test data corpus (with the MusicXML files)
and the target directory. The images will be then exported into the target
directory, so that they can be imported into the database afterwards.

Relies on MuseScore being available to convert MusicXML files to PNG.
Uses the crop_image_for_omreval.py script to crop the A4-sized PNG outputs
of MuseScore to something more usable: a bounding box around everything
not transparent, plus a 10-pixel margin.

Options
-------

  -d        Test data root directory. Assumes all XML files are MusicXML
            files that should be exported for the database. The directory
            structure is preserved.

  -t        Target directory. Will be created if it does not exist.

  -M        [Path to] MuseScore executable.

  -O        If target directory exists, warn, remove and overwrite.
            (If -O is not set, will exit.)
USAGE
)

selfdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${selfdir}/../../util/SPUtils.sh

##############################################################################

DATA_DIR=
TARGET_DIR=
OVERWRITE=
MSCORE="/Applications/MuseScore2.app/Contents/MacOS/mscore"

while getopts "hd:t:M:O" opt; do
    case ${opt} in
        h) echo "${usage}"; exit 0;;
        t) TARGET_DIR=${OPTARG};;
        d) DATA_DIR=${OPTARG};;
        O) OVERWRITE=1;;
        M) MSCORE=${OPTARG};;
        \?) echo "Invalid option: ${OPTARG}"; echo ${usage};;
    esac
done

SPCheckDir "${DATA_DIR}"
SPCheckExecWithSpaces "${MSCORE}"

if [ ! -d ${TARGET_DIR} ]; then
    mkdir -p ${TARGET_DIR}
elif [ ! -z ${OVERWRITE} ]; then
    SPLogWarning "Target directory ${TARGET_DIR} already exists!" >&2
    SPLogWarning "Overwriting in 5 seconds (ctrl+c to stop)." >&2
    sleep 5
    rm -r ${TARGET_DIR}
    mkdir -p ${TARGET_DIR}
else
    SPLogWarning "Target directory ${TARGET_DIR} already exists!" >&2
    SPLogError "Overwrite not set, aborting." >&2
fi

##############################################################################

# rsync the XML files to target dir
rsync rsync -avm --include="*/" --include="**/*.xml" --exclude="*" ${DATA_DIR} ${TARGET_DIR}

# Convert each to png
for xmlf in `find ${TARGET_DIR} -name *.xml`; do
    dir=`dirname ${xmlf}`
    fname=`basename ${xmlf}`
    ofname=`echo ${fname} | sed 's/.xml$/.png/'`
    target=${dir}/${ofname}

    SPCheckFile ${xmlf}

    echo -e "\n===================================="
    SPLogInfo "Input file:  ${xmlf}"
    SPLogInfo "Target file: ${target}"
    "${MSCORE}" ${xmlf} -o ${target}

    SPLogInfo "MuseScore finished conversion."

    # Postprocessing MuseScore png filename: remove the "-1"
    # (But don't rename, just use it as output name for cropping.)
    actual_ofname=`echo ${target} | sed 's/.png$/-1.png/'`
    SPLogInfo "Checking if MScore output file exists: ${actual_ofname}"
    SPCheckFile ${actual_ofname}
    rm ${xmlf}

    # Crop. For now, verbose.
    SPLogInfo "Cropping..."
    ${selfdir}/crop_image_for_omreval_server.py -i ${actual_ofname} -o ${target} --margin 10 -v

    SPLogInfo "Checking if crop output file exists: ${target}"
    SPCheckFile ${target}
    rm ${actual_ofname}
    SPLogInfo "File done!"
done


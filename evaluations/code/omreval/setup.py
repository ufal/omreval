from setuptools import setup

setup(
    name='omreval',
    version='1.0.0',
    package_dir={'omreval': 'omreval'},
    packages=['omreval'],
    scripts=['omreval/crop_image_for_omreval_server.py',
             'omreval/export_images_for_omreval_server.sh',
             'omreval/levenshtein_eval.py',
             'omreval/lilypond_eval.py',
             'omreval/treedist_eval.py'],
    url='',
    license='Institute of Formal and Applied Linguistics, Charles University in Prague. All rights reserved.',
    author='Jan Hajic jr.',
    author_email='hajicj@ufal.mff.cuni.cz',
    description='Modules and scripts for evaluating Optical Music Recognition.'
)
